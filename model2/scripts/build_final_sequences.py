from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

BASE = Path("/home/administrator/24DIT065/model2")

DATA = BASE / "04_model/datasets/multimodal_dataset_daily_clean.parquet"
IMPACT = BASE / "04_model/datasets/news_impact_predictions_v2.parquet"

OUT = BASE / "04_model/sequences"
OUT.mkdir(parents=True, exist_ok=True)

LOOKBACK = 60

TRAIN_END = pd.Timestamp("2023-12-31")
VAL_END = pd.Timestamp("2025-12-31")


print("=" * 110)
print("MODEL 2 — FINAL 60-DAY MULTIMODAL SEQUENCES")
print("=" * 110)


# =============================================================================
# LOAD
# =============================================================================

df = pd.read_parquet(DATA)
impact = pd.read_parquet(IMPACT)

df["Date"] = pd.to_datetime(df["Date"])
impact["Date"] = pd.to_datetime(impact["Date"])

df = df.sort_values(
    ["symbol", "Date"]
).reset_index(drop=True)

impact = impact[
    [
        "Date",
        "symbol",
        "impact_prediction_source",
    ]
].drop_duplicates(
    ["Date", "symbol"]
)


# =============================================================================
# ADD AVAILABILITY FLAGS
# =============================================================================

df = df.merge(
    impact,
    on=["Date", "symbol"],
    how="left",
)

# News availability:
# news_count > 0 means matched news was observed for this stock/date.

if "news_news_count" in df.columns:
    df["news_available"] = (
        df["news_news_count"] > 0
    ).astype(np.float32)
else:
    df["news_available"] = 0.0


# Impact availability:
# true only when a leakage-safe impact prediction exists.

df["impact_available"] = (
    df["impact_prediction_source"]
    .fillna("unavailable")
    .ne("unavailable")
    .astype(np.float32)
)

df = df.drop(
    columns=["impact_prediction_source"]
)

print(
    "\nNews available:",
    int(df["news_available"].sum()),
)

print(
    "Impact available:",
    int(df["impact_available"].sum()),
)


# =============================================================================
# TARGETS
# =============================================================================

target_cols = [
    "target_return_1d",
    "target_return_3d",
    "target_return_5d",
    "target_direction_1d",
    "target_direction_5d",
]


# =============================================================================
# FEATURE COLUMNS
# =============================================================================

feature_cols = [
    c for c in df.columns
    if c not in (
        ["Date", "symbol"] +
        target_cols
    )
    and pd.api.types.is_numeric_dtype(df[c])
]

# Explicitly ensure availability flags are included.
for c in [
    "news_available",
    "impact_available",
]:
    if c not in feature_cols:
        feature_cols.append(c)


# Keep deterministic order.
feature_cols = sorted(set(feature_cols))

print("\nFeature count:", len(feature_cols))

print("\nFirst 30 features:")
for c in feature_cols[:30]:
    print(" ", c)


# =============================================================================
# SPLIT ENDPOINTS
# =============================================================================

df["split"] = np.select(
    [
        df["Date"] <= TRAIN_END,
        df["Date"] <= VAL_END,
    ],
    [
        "train",
        "validation",
    ],
    default="test",
)


# =============================================================================
# TRAIN-ONLY ROBUST SCALING
# =============================================================================
#
# Fit on TRAIN rows only.
#
# We use RobustScaler because financial data contains heavy tails and
# occasional extreme observations.
#
# Scaling is learned only from the training period.
# =============================================================================

train_rows = df["split"] == "train"

X_train_raw = (
    df.loc[
        train_rows,
        feature_cols
    ]
    .replace([np.inf, -np.inf], np.nan)
)

# Training medians first.
training_medians = (
    X_train_raw
    .median()
)

X_train_raw = (
    X_train_raw
    .fillna(training_medians)
)

scaler = RobustScaler(
    quantile_range=(25, 75)
)

scaler.fit(
    X_train_raw
)

# Save scaling parameters for inference.
scale_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "median": training_medians.values,
        "center": scaler.center_,
        "scale": scaler.scale_,
    }
)

scale_df.to_csv(
    OUT / "training_robust_scaler.csv",
    index=False,
)

# ---------------------------------------------------------------------
# APPLY SCALING
# ---------------------------------------------------------------------

X_all = (
    df[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(training_medians)
)

X_scaled = scaler.transform(
    X_all
).astype(np.float32)

# Avoid pathological extreme scaled values.
X_scaled = np.clip(
    X_scaled,
    -10.0,
    10.0,
).astype(np.float32)

print("\nScaled matrix:")
print(X_scaled.shape)

print(
    "Scaled finite:",
    np.isfinite(X_scaled).all()
)


# =============================================================================
# TARGET MATRIX
# =============================================================================

Y = (
    df[target_cols]
    .to_numpy(dtype=np.float32)
)

print(
    "Targets finite:",
    np.isfinite(Y).all()
)


# =============================================================================
# STOCK INDEX
# =============================================================================

symbols = sorted(
    df["symbol"].unique()
)

symbol_to_id = {
    s: i
    for i, s in enumerate(symbols)
}

stock_id = (
    df["symbol"]
    .map(symbol_to_id)
    .to_numpy(
        dtype=np.int32
    )
)

pd.DataFrame(
    {
        "stock_id": range(len(symbols)),
        "symbol": symbols,
    }
).to_csv(
    OUT / "stock_mapping.csv",
    index=False,
)


# =============================================================================
# DATE INDEX
# =============================================================================

dates = df["Date"].to_numpy()


# =============================================================================
# BUILD SEQUENCES
# =============================================================================
#
# IMPORTANT:
#
# Each sequence is generated separately per stock.
#
# Sequence endpoint = prediction date.
#
# For validation/test endpoints, historical observations before the
# split boundary are allowed because they were known at that time.
#
# We never use observations after the endpoint.
# =============================================================================

def build_split_sequences(split_name):

    endpoint_mask = (
        df["split"].values == split_name
    )

    endpoint_indices = np.where(
        endpoint_mask
    )[0]

    X_list = []
    y_1d = []
    y_3d = []
    y_5d = []
    dir_1d = []
    dir_5d = []
    date_list = []
    stock_list = []

    # Work by stock to guarantee contiguous 60-session history.
    for symbol in symbols:

        idx = np.where(
            df["symbol"].values == symbol
        )[0]

        # idx is chronological because df was sorted.
        split_idx = idx[
            endpoint_mask[idx]
        ]

        if len(split_idx) == 0:
            continue

        # Position lookup: global dataframe index -> position inside stock array.
        global_to_pos = {
            int(g): p
            for p, g in enumerate(idx)
        }

        for global_end in split_idx:

            pos = global_to_pos[int(global_end)]

            if pos < LOOKBACK - 1:
                continue

            start = pos - LOOKBACK + 1

            window_global = idx[
                start:pos + 1
            ]

            # Exact lookback.
            if len(window_global) != LOOKBACK:
                continue

            seq = X_scaled[
                window_global
            ]

            # Final point target.
            target_row = global_end

            if not np.isfinite(seq).all():
                continue

            target = Y[
                target_row
            ]

            if not np.isfinite(target).all():
                continue

            X_list.append(seq)

            y_1d.append(target[0])
            y_3d.append(target[1])
            y_5d.append(target[2])

            dir_1d.append(target[3])
            dir_5d.append(target[4])

            date_list.append(
                dates[target_row]
            )

            stock_list.append(
                stock_id[target_row]
            )

    if not X_list:
        raise RuntimeError(
            f"No sequences created for split={split_name}"
        )

    X = np.stack(
        X_list
    ).astype(np.float32)

    y1 = np.asarray(
        y_1d,
        dtype=np.float32
    )

    y3 = np.asarray(
        y_3d,
        dtype=np.float32
    )

    y5 = np.asarray(
        y_5d,
        dtype=np.float32
    )

    d1 = np.asarray(
        dir_1d,
        dtype=np.float32
    )

    d5 = np.asarray(
        dir_5d,
        dtype=np.float32
    )

    dates_arr = np.asarray(
        date_list
    )

    stock_arr = np.asarray(
        stock_list,
        dtype=np.int32
    )

    return (
        X,
        y1,
        y3,
        y5,
        d1,
        d5,
        dates_arr,
        stock_arr,
    )


# =============================================================================
# CREATE EACH SPLIT
# =============================================================================

for split_name in [
    "train",
    "validation",
    "test",
]:

    print(
        f"\nBuilding {split_name} sequences..."
    )

    (
        X,
        y1,
        y3,
        y5,
        d1,
        d5,
        dates_arr,
        stock_arr,
    ) = build_split_sequences(
        split_name
    )

    prefix = {
        "train": "train_60d",
        "validation": "val_60d",
        "test": "test_60d",
    }[split_name]

    np.save(
        OUT / f"{prefix}_X.npy",
        X,
    )

    np.save(
        OUT / f"{prefix}_y_return_1d.npy",
        y1,
    )

    np.save(
        OUT / f"{prefix}_y_return_3d.npy",
        y3,
    )

    np.save(
        OUT / f"{prefix}_y_return_5d.npy",
        y5,
    )

    np.save(
        OUT / f"{prefix}_y_direction_1d.npy",
        d1,
    )

    np.save(
        OUT / f"{prefix}_y_direction_5d.npy",
        d5,
    )

    np.save(
        OUT / f"{prefix}_dates.npy",
        dates_arr,
    )

    np.save(
        OUT / f"{prefix}_stock_id.npy",
        stock_arr,
    )

    print(
        f"{split_name}: "
        f"X={X.shape}, "
        f"y1={y1.shape}, "
        f"y5={y5.shape}"
    )

    print(
        f"Date range: "
        f"{pd.Timestamp(dates_arr.min())} -> "
        f"{pd.Timestamp(dates_arr.max())}"
    )


# =============================================================================
# FINAL MANIFEST
# =============================================================================

manifest = pd.DataFrame(
    {
        "feature_index": range(len(feature_cols)),
        "feature": feature_cols,
    }
)

manifest.to_csv(
    OUT / "final_sequence_feature_manifest.csv",
    index=False,
)

print("\n" + "=" * 110)
print("FINAL SEQUENCE BUILD COMPLETE")
print("=" * 110)

print("Lookback:", LOOKBACK)
print("Features:", len(feature_cols))
print("Stocks:", len(symbols))

print("\nSaved to:")
print(OUT)

print("\nSTATUS: PASS — FINAL 60-DAY MULTIMODAL SEQUENCES READY")
print("=" * 110)
