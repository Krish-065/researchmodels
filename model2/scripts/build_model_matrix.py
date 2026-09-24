from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.home() / "24DIT065" / "model2"

SPLIT_DIR = ROOT / "04_model" / "splits"
OUT_DIR = ROOT / "04_model" / "datasets"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD SPLITS
# ============================================================

train = pd.read_parquet(SPLIT_DIR / "train.parquet")
valid = pd.read_parquet(SPLIT_DIR / "validation.parquet")
test = pd.read_parquet(SPLIT_DIR / "test.parquet")


# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    "target_return_1d",
    "target_return_3d",
    "target_return_5d",
    "target_direction_1d",
    "target_direction_5d",
]


# ============================================================
# NON-FEATURE COLUMNS
# ============================================================

NON_FEATURES = {
    "Date",
    "symbol",

    # Raw market fields
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",

    # Targets
    *TARGETS,
}


# ============================================================
# CANDIDATE FEATURES
# ============================================================

candidate = [
    c for c in train.columns
    if c not in NON_FEATURES
]


# ============================================================
# FEATURE GROUPS
# ============================================================

# Stock technical features.
stock_features = [
    c for c in candidate
    if not any(
        c.startswith(prefix + "_")
        for prefix in [
            "nifty50",
            "indiavix",
            "sp500",
            "nasdaq",
            "dow",
            "nikkei",
            "hangseng",
            "gold",
            "crude",
            "usdinr",
            "btc",
            "eth",
        ]
    )
]


# Related asset features.
related_prefixes = [
    "nifty50",
    "indiavix",
    "sp500",
    "nasdaq",
    "dow",
    "nikkei",
    "hangseng",
    "gold",
    "crude",
    "usdinr",
    "btc",
    "eth",
]

related_features = [
    c for c in candidate
    if any(
        c.startswith(prefix + "_")
        for prefix in related_prefixes
    )
]


# ============================================================
# REMOVE VERY SPARSE FEATURES
# ============================================================

# Fit the missingness rule using TRAIN ONLY.
train_missing_rate = train[candidate].isna().mean()

MAX_MISSING_RATE = 0.30

keep_sparse = train_missing_rate[
    train_missing_rate <= MAX_MISSING_RATE
].index.tolist()


# ============================================================
# SELECT FEATURE SET
# ============================================================

features = [
    c for c in candidate
    if c in keep_sparse
]


# Explicitly exclude raw related prices.
features = [
    c for c in features
    if not (
        c.endswith("_close")
        or c.endswith("_adj_close")
    )
]


# ============================================================
# SAFETY CHECK
# ============================================================

for target in TARGETS:
    assert target not in features

assert "Date" not in features
assert "symbol" not in features

assert len(features) > 50


print("=" * 100)
print("MODEL 2 — BUILD MODEL MATRIX")
print("=" * 100)

print("Candidate features :", len(candidate))
print("Selected features  :", len(features))
print("Removed sparse     :", len(candidate) - len(features))

print()
print("Stock feature count :", sum(c in stock_features for c in features))
print(
    "Related feature count:",
    sum(c in related_features for c in features)
)


# ============================================================
# TRAIN-ONLY IMPUTATION
# ============================================================

print()
print("Computing training-set medians...")

medians = (
    train[features]
    .replace([np.inf, -np.inf], np.nan)
    .median()
)


# Any feature completely missing in train should not survive.
valid_features = [
    c for c in features
    if pd.notna(medians[c])
]

features = valid_features

print("Features after median check:", len(features))


def prepare(df):

    X = (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )

    X = X.fillna(medians[features])

    # Final numerical safety.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0.0)

    return X


X_train = prepare(train)
X_valid = prepare(valid)
X_test = prepare(test)


# ============================================================
# SAVE MATRICES
# ============================================================

X_train.to_parquet(
    OUT_DIR / "X_train.parquet",
    index=False
)

X_valid.to_parquet(
    OUT_DIR / "X_validation.parquet",
    index=False
)

X_test.to_parquet(
    OUT_DIR / "X_test.parquet",
    index=False
)


# Save targets separately.
for name, df in [
    ("train", train),
    ("validation", valid),
    ("test", test),
]:

    df[
        [
            "Date",
            "symbol",
            *TARGETS,
        ]
    ].to_parquet(
        OUT_DIR / f"y_{name}.parquet",
        index=False
    )


# Save feature names.
pd.Series(features, name="feature").to_csv(
    OUT_DIR / "feature_list.csv",
    index=False
)


# Save training medians.
medians[features].to_csv(
    OUT_DIR / "training_medians.csv",
    header=["median"]
)


# ============================================================
# AUDIT
# ============================================================

print()
print("=" * 100)
print("MODEL MATRIX CREATED")
print("=" * 100)

print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("X_test :", X_test.shape)

print()
print("Remaining missing values:")

print("train:", X_train.isna().sum().sum())
print("valid:", X_valid.isna().sum().sum())
print("test :", X_test.isna().sum().sum())

print()
print("Remaining infinite values:")

print("train:", np.isinf(X_train.to_numpy()).sum())
print("valid:", np.isinf(X_valid.to_numpy()).sum())
print("test :", np.isinf(X_test.to_numpy()).sum())

print()
print("Saved feature list:")
print(OUT_DIR / "feature_list.csv")

print()
print("=" * 100)

assert X_train.shape[1] == X_valid.shape[1]
assert X_train.shape[1] == X_test.shape[1]

assert X_train.isna().sum().sum() == 0
assert X_valid.isna().sum().sum() == 0
assert X_test.isna().sum().sum() == 0

assert np.isinf(X_train.to_numpy()).sum() == 0
assert np.isinf(X_valid.to_numpy()).sum() == 0
assert np.isinf(X_test.to_numpy()).sum() == 0

print("STATUS: PASS — MODEL MATRIX READY")
print("=" * 100)
