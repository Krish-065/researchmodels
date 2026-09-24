from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE = Path("/home/administrator/24DIT065/model2")

LABEL_FILE = BASE / "04_model/datasets/news_impact_labels_v2.parquet"
NEWS_FILE = BASE / "03_features/news/news_daily_stock_features.parquet"
MARKET_FILE = BASE / "03_features/market/stock_features_daily.parquet"

OUT_DIR = BASE / "04_model/datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_FILE = OUT_DIR / "news_impact_predictions_v2.parquet"
METRICS_FILE = OUT_DIR / "news_impact_v2_model_metrics.csv"

print("=" * 110)
print("MODEL 2 — NEWS IMPACT V2 PREDICTOR")
print("Continuous regression + chronological OOF predictions")
print("=" * 110)

# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------

labels = pd.read_parquet(LABEL_FILE)
news = pd.read_parquet(NEWS_FILE)
market = pd.read_parquet(MARKET_FILE)

for df in [labels, news, market]:
    df["Date"] = pd.to_datetime(df["Date"])

# ---------------------------------------------------------------------
# NEWS FEATURES
# ---------------------------------------------------------------------

news_cols = [
    "Date",
    "symbol",
    "news_count",
    "news_mean_sentiment",
    "news_std_sentiment",
    "news_max_positive",
    "news_min_negative",
    "news_positive_count",
    "news_negative_count",
    "news_neutral_count",
    "news_count_3d",
    "news_sentiment_3d",
    "news_count_5d",
    "news_sentiment_5d",
    "news_count_10d",
    "news_sentiment_10d",
    "news_intensity",
    "news_sentiment_pressure",
]

news = news[
    [c for c in news_cols if c in news.columns]
].copy()

# ---------------------------------------------------------------------
# PRE-EVENT MARKET FEATURES
#
# Shift one trading day so the model cannot see same-day reaction.
# ---------------------------------------------------------------------

candidate_market = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "log_return_1d",
    "close_vs_sma_20",
    "close_vs_sma_50",
    "close_vs_sma_200",
    "atr_pct",
    "high_low_range",
    "open_close_range",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_50d",
    "volume_change_1d",
    "volume_ratio_20",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "momentum_50",
    "distance_high_20",
    "distance_low_20",
    "distance_high_50",
    "distance_low_50",
    "distance_high_200",
    "distance_low_200",
]

available_market = [
    c for c in candidate_market
    if c in market.columns
]

market = market[
    ["Date", "symbol"] + available_market
].copy()

market = (
    market
    .sort_values(["symbol", "Date"])
    .reset_index(drop=True)
)

for c in available_market:
    market[f"prev_{c}"] = (
        market.groupby("symbol")[c].shift(1)
    )

market = market[
    ["Date", "symbol"] +
    [f"prev_{c}" for c in available_market]
]

# ---------------------------------------------------------------------
# MERGE
# ---------------------------------------------------------------------

df = labels[
    [
        "Date",
        "symbol",
        "realized_impact_score_v2",
        "impact_class_v2_id",
    ]
].merge(
    news,
    on=["Date", "symbol"],
    how="inner",
).merge(
    market,
    on=["Date", "symbol"],
    how="left",
)

df = df.sort_values(
    ["Date", "symbol"]
).reset_index(drop=True)

feature_cols = [
    c for c in df.columns
    if c.startswith("news_") or c.startswith("prev_")
]

feature_cols = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(df[c])
]

print(f"Rows: {len(df):,}")
print(f"Features: {len(feature_cols)}")

# ---------------------------------------------------------------------
# TEMPORAL SPLIT
# ---------------------------------------------------------------------

train_mask = df["Date"] <= pd.Timestamp("2023-12-31")

val_mask = (
    (df["Date"] > pd.Timestamp("2023-12-31")) &
    (df["Date"] <= pd.Timestamp("2025-12-31"))
)

test_mask = df["Date"] > pd.Timestamp("2025-12-31")

train = df.loc[train_mask].copy()
val = df.loc[val_mask].copy()
test = df.loc[test_mask].copy()

X_train = train[feature_cols].replace(
    [np.inf, -np.inf], np.nan
)
X_val = val[feature_cols].replace(
    [np.inf, -np.inf], np.nan
)
X_test = test[feature_cols].replace(
    [np.inf, -np.inf], np.nan
)

y_train = train["realized_impact_score_v2"].astype(float)
y_val = val["realized_impact_score_v2"].astype(float)
y_test = test["realized_impact_score_v2"].astype(float)

# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

def make_model():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )

# ---------------------------------------------------------------------
# CHRONOLOGICAL OOF TRAIN PREDICTIONS
# ---------------------------------------------------------------------

n = len(train)
edges = np.linspace(0, n, 6, dtype=int)

oof = np.full(n, np.nan)

print("\nChronological OOF:")

for fold in range(1, 6):
    pred_start = edges[fold]
    pred_end = edges[fold + 1] if fold < 5 else n

    fit_idx = np.arange(0, pred_start)
    pred_idx = np.arange(pred_start, pred_end)

    if len(fit_idx) < 100 or len(pred_idx) == 0:
        continue

    model = make_model()

    model.fit(
        X_train.iloc[fit_idx],
        y_train.iloc[fit_idx],
    )

    oof[pred_idx] = model.predict(
        X_train.iloc[pred_idx]
    )

    print(
        f"Fold {fold}: "
        f"fit={len(fit_idx):,} "
        f"pred={len(pred_idx):,}"
    )

# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

model_train = make_model()

model_train.fit(
    X_train,
    y_train,
)

val_pred = model_train.predict(X_val)

# ---------------------------------------------------------------------
# TEST
#
# Strict procedure:
# test is predicted only after fitting on train + validation.
# ---------------------------------------------------------------------

X_train_val = pd.concat(
    [X_train, X_val],
    axis=0,
)

y_train_val = pd.concat(
    [y_train, y_val],
    axis=0,
)

model_final = make_model()

model_final.fit(
    X_train_val,
    y_train_val,
)

test_pred = model_final.predict(X_test)

# ---------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------

for name, arr in [
    ("oof", oof),
    ("validation", val_pred),
    ("test", test_pred),
]:
    mask = np.isfinite(arr)

    if mask.any():
        arr[mask] = np.clip(
            arr[mask],
            0.0,
            1.0,
        )

# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def regression_metrics(name, y, pred):
    mask = np.isfinite(pred)

    yt = np.asarray(y)[mask]
    yp = np.asarray(pred)[mask]

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(
        mean_squared_error(yt, yp)
    )

    corr = np.corrcoef(yt, yp)[0, 1]

    return {
        "split": name,
        "rows": len(yt),
        "mae": mae,
        "rmse": rmse,
        "pearson_corr": corr,
    }

metrics = []

oof_mask = np.isfinite(oof)

if oof_mask.any():
    metrics.append(
        regression_metrics(
            "train_oof",
            y_train,
            oof,
        )
    )

metrics.append(
    regression_metrics(
        "validation",
        y_val,
        val_pred,
    )
)

metrics.append(
    regression_metrics(
        "test",
        y_test,
        test_pred,
    )
)

metrics_df = pd.DataFrame(metrics)

print("\n" + "=" * 110)
print("V2 IMPACT MODEL METRICS")
print("=" * 110)

print(
    metrics_df.to_string(index=False)
)

# ---------------------------------------------------------------------
# BUILD PREDICTION DATASET
# ---------------------------------------------------------------------

train_out = train[
    ["Date", "symbol"]
].copy()

train_out[
    "predicted_news_impact_score"
] = oof

train_out[
    "impact_prediction_source"
] = np.where(
    np.isfinite(oof),
    "chronological_oof",
    "unavailable",
)

val_out = val[
    ["Date", "symbol"]
].copy()

val_out[
    "predicted_news_impact_score"
] = val_pred

val_out[
    "impact_prediction_source"
] = "train_fitted"

test_out = test[
    ["Date", "symbol"]
].copy()

test_out[
    "predicted_news_impact_score"
] = test_pred

test_out[
    "impact_prediction_source"
] = "train_validation_fitted"

out = pd.concat(
    [
        train_out,
        val_out,
        test_out,
    ],
    ignore_index=True,
)

# ---------------------------------------------------------------------
# CONVERT CONTINUOUS SCORE TO SOFT IMPACT LEVELS
#
# IMPORTANT:
# Some earliest chronological OOF rows intentionally have NaN because
# there is no earlier training history available to generate a leakage-safe
# prediction. Those rows remain unavailable rather than being forced into
# a class.
# ---------------------------------------------------------------------

score = out[
    "predicted_news_impact_score"
].astype(float)

temperature = 0.12

centers = np.array(
    [0.18, 0.36, 0.61],
    dtype=float,
)

values = score.to_numpy()

probs = np.full(
    (len(out), 3),
    np.nan,
    dtype=float,
)

valid_score = np.isfinite(values)

if valid_score.any():

    valid_values = values[valid_score]

    logits = -(
        (valid_values[:, None] - centers[None, :]) ** 2
    ) / temperature

    # Stable softmax
    logits_max = np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exp_logits = np.exp(
        logits - logits_max
    )

    denom = exp_logits.sum(
        axis=1,
        keepdims=True,
    )

    probs[valid_score] = (
        exp_logits / denom
    )

out[
    "news_impact_low_prob"
] = probs[:, 0]

out[
    "news_impact_medium_prob"
] = probs[:, 1]

out[
    "news_impact_high_prob"
] = probs[:, 2]

out["predicted_news_impact"] = np.nan

if valid_score.any():

    valid_classes = np.argmax(
        probs[valid_score],
        axis=1,
    )

    out.loc[
        valid_score,
        "predicted_news_impact"
    ] = valid_classes

out["predicted_news_impact_label"] = (
    out["predicted_news_impact"]
    .map(
        {
            0: "LOW",
            1: "MEDIUM",
            2: "HIGH",
        }
    )
)

# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

out = out.sort_values(
    ["Date", "symbol"]
).reset_index(drop=True)

out.to_parquet(
    PRED_FILE,
    index=False,
    compression="snappy",
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False,
)

print(f"\nSaved: {PRED_FILE}")
print(f"Saved: {METRICS_FILE}")

print("\nPrediction source:")
print(
    out["impact_prediction_source"]
    .value_counts()
    .to_string()
)

print("\nPredicted labels:")
print(
    out["predicted_news_impact_label"]
    .value_counts(dropna=False)
    .to_string()
)

print("\n" + "=" * 110)
print("STATUS: PASS — NEWS IMPACT V2 PREDICTOR COMPLETE")
print("=" * 110)
# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

out = out.sort_values(
    ["Date", "symbol"]
).reset_index(drop=True)

out.to_parquet(
    PRED_FILE,
    index=False,
    compression="snappy",
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False,
)

print(f"\nSaved: {PRED_FILE}")
print(f"Saved: {METRICS_FILE}")

print("\nPrediction source:")
print(
    out["impact_prediction_source"]
    .value_counts()
    .to_string()
)

print("\nPredicted labels:")
print(
    out["predicted_news_impact_label"]
    .value_counts(dropna=False)
    .to_string()
)

print("\n" + "=" * 110)
print("STATUS: PASS — NEWS IMPACT V2 PREDICTOR COMPLETE")
print("=" * 110)
