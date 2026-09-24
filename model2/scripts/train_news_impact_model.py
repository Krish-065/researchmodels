from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =============================================================================
# PATHS
# =============================================================================

BASE = Path("/home/administrator/24DIT065/model2")

LABEL_FILE = BASE / "04_model/datasets/news_impact_labels.parquet"
NEWS_FILE = BASE / "03_features/news/news_daily_stock_features.parquet"
MARKET_FILE = BASE / "03_features/market/stock_features_daily.parquet"

OUT_DIR = BASE / "04_model/datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "news_impact_predictions.parquet"
METRICS_FILE = OUT_DIR / "news_impact_model_metrics.csv"


print("=" * 110)
print("MODEL 2 — LEAKAGE-SAFE NEWS IMPACT CLASSIFIER")
print("=" * 110)


# =============================================================================
# LOAD
# =============================================================================

labels = pd.read_parquet(LABEL_FILE)
news = pd.read_parquet(NEWS_FILE)
market = pd.read_parquet(MARKET_FILE)

labels["Date"] = pd.to_datetime(labels["Date"])
news["Date"] = pd.to_datetime(news["Date"])
market["Date"] = pd.to_datetime(market["Date"])

print(f"Impact labels : {len(labels):,}")
print(f"News rows     : {len(news):,}")
print(f"Market rows   : {len(market):,}")


# =============================================================================
# PREPARE NEWS FEATURES
# =============================================================================

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

news = news[[c for c in news_cols if c in news.columns]].copy()


# =============================================================================
# MARKET FEATURES
# =============================================================================
#
# CRITICAL:
# We do NOT use same-day realized market reaction.
#
# Every market predictor is shifted by one trading session within each stock.
# Thus a news event dated D only sees information known before D.
#
# =============================================================================

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
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_position",
    "bb_width",
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

market = market.sort_values(
    ["symbol", "Date"]
).reset_index(drop=True)

# Shift all market features by one trading session.
for c in available_market:
    market[c] = market.groupby("symbol")[c].shift(1)

market = market.rename(
    columns={c: f"prev_{c}" for c in available_market}
)


# =============================================================================
# MERGE
# =============================================================================

df = labels[
    ["Date", "symbol", "impact_class_id"]
].merge(
    news,
    on=["Date", "symbol"],
    how="left",
)

df = df.merge(
    market,
    on=["Date", "symbol"],
    how="left",
)

df = df.sort_values(
    ["Date", "symbol"]
).reset_index(drop=True)

print(f"Merged impact dataset: {len(df):,}")


# =============================================================================
# FEATURE SELECTION
# =============================================================================

news_feature_cols = [
    c for c in news.columns
    if c not in ["Date", "symbol"]
]

market_feature_cols = [
    c for c in df.columns
    if c.startswith("prev_")
]

feature_cols = news_feature_cols + market_feature_cols

# Drop accidental non-numeric columns
feature_cols = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(df[c])
]

print(f"Feature count: {len(feature_cols)}")


# =============================================================================
# TEMPORAL SPLIT
# =============================================================================

TRAIN_END = pd.Timestamp("2023-12-31")
VAL_END = pd.Timestamp("2025-12-31")

train_mask = df["Date"] <= TRAIN_END
val_mask = (
    (df["Date"] > TRAIN_END) &
    (df["Date"] <= VAL_END)
)
test_mask = df["Date"] > VAL_END

train = df.loc[train_mask].copy()
val = df.loc[val_mask].copy()
test = df.loc[test_mask].copy()

print("\nTemporal split:")
print(
    f"Train: {len(train):,} | "
    f"{train.Date.min()} -> {train.Date.max()}"
)
print(
    f"Validation: {len(val):,} | "
    f"{val.Date.min()} -> {val.Date.max()}"
)
print(
    f"Test: {len(test):,} | "
    f"{test.Date.min()} -> {test.Date.max()}"
)


# =============================================================================
# MATRIX
# =============================================================================

X_train = train[feature_cols].replace(
    [np.inf, -np.inf], np.nan
)
X_val = val[feature_cols].replace(
    [np.inf, -np.inf], np.nan
)
X_test = test[feature_cols].replace(
    [np.inf, -np.inf], np.nan
)

y_train = train["impact_class_id"].astype(int)
y_val = val["impact_class_id"].astype(int)
y_test = test["impact_class_id"].astype(int)


# =============================================================================
# MODEL
# =============================================================================

def make_model():
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
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


# =============================================================================
# OUT-OF-FOLD PREDICTIONS FOR TRAIN
# =============================================================================
#
# This prevents the final trading model from seeing impact probabilities
# generated by a classifier that trained directly on the exact same row.
#
# Five chronological folds.
# =============================================================================

n_train = len(train)

fold_edges = np.linspace(
    0,
    n_train,
    6,
    dtype=int,
)

train_oof = np.full(
    (n_train, 3),
    np.nan,
    dtype=float,
)

print("\nGenerating chronological OOF probabilities...")

for fold in range(1, 6):

    fit_end = fold_edges[fold]
    pred_start = fold_edges[fold]
    pred_end = fold_edges[fold + 1] if fold < 5 else n_train

    if pred_start >= pred_end:
        continue

    # Need a strictly prior fitting set
    fit_end = fold_edges[fold]

    if fit_end <= 0:
        continue

    fit_idx = np.arange(0, pred_start)
    pred_idx = np.arange(pred_start, pred_end)

    if len(fit_idx) < 100:
        continue

    model = make_model()

    model.fit(
        X_train.iloc[fit_idx],
        y_train.iloc[fit_idx],
    )

    train_oof[pred_idx] = model.predict_proba(
        X_train.iloc[pred_idx]
    )

    print(
        f"Fold {fold}: "
        f"fit={len(fit_idx):,}, "
        f"predict={len(pred_idx):,}"
    )

# Rows before first prediction fold remain NaN.
# Fill them later from a model trained only on earlier data is impossible.
# We will leave them as NaN and use the full-train fitted model only for the
# downstream training matrix if needed.
#
# For evaluation, only OOF-covered rows are used.


# =============================================================================
# FULL TRAIN MODEL
# =============================================================================

model_train = make_model()

model_train.fit(
    X_train,
    y_train,
)

val_prob = model_train.predict_proba(X_val)

# Freeze train/validation methodology before touching test.
# For final test prediction, refit on TRAIN + VALIDATION.
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

test_prob = model_final.predict_proba(X_test)


# =============================================================================
# METRICS
# =============================================================================

def evaluate(name, y_true, prob):

    pred = np.argmax(prob, axis=1)

    return {
        "split": name,
        "rows": len(y_true),
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(
            y_true,
            pred,
            average="macro",
        ),
        "weighted_f1": f1_score(
            y_true,
            pred,
            average="weighted",
        ),
        "log_loss": log_loss(
            y_true,
            prob,
            labels=[0, 1, 2],
        ),
    }


metrics = []

# OOF evaluation
oof_mask = np.isfinite(train_oof).all(axis=1)

if oof_mask.sum() > 0:
    metrics.append(
        evaluate(
            "train_oof",
            y_train.iloc[np.where(oof_mask)[0]],
            train_oof[oof_mask],
        )
    )

metrics.append(
    evaluate(
        "validation",
        y_val,
        val_prob,
    )
)

metrics.append(
    evaluate(
        "test",
        y_test,
        test_prob,
    )
)

metrics_df = pd.DataFrame(metrics)

print("\n" + "=" * 110)
print("IMPACT MODEL METRICS")
print("=" * 110)

print(metrics_df.to_string(index=False))


# =============================================================================
# TEST REPORT
# =============================================================================

test_pred = np.argmax(test_prob, axis=1)

print("\nTest classification report:")
print(
    classification_report(
        y_test,
        test_pred,
        target_names=["LOW", "MEDIUM", "HIGH"],
        digits=4,
    )
)

print("Test confusion matrix:")
print(
    confusion_matrix(
        y_test,
        test_pred,
    )
)


# =============================================================================
# BUILD OUTPUT
# =============================================================================

train_out = train[
    ["Date", "symbol"]
].copy()

# Use OOF where available
train_out["news_impact_low_prob"] = train_oof[:, 0]
train_out["news_impact_medium_prob"] = train_oof[:, 1]
train_out["news_impact_high_prob"] = train_oof[:, 2]

# For early train rows not covered by OOF, we deliberately do NOT leak
# in-sample predictions. Mark them as unavailable.
train_out["impact_prediction_source"] = np.where(
    np.isfinite(train_oof).all(axis=1),
    "chronological_oof",
    "unavailable",
)

val_out = val[
    ["Date", "symbol"]
].copy()

val_out["news_impact_low_prob"] = val_prob[:, 0]
val_out["news_impact_medium_prob"] = val_prob[:, 1]
val_out["news_impact_high_prob"] = val_prob[:, 2]
val_out["impact_prediction_source"] = "train_fitted"

test_out = test[
    ["Date", "symbol"]
].copy()

test_out["news_impact_low_prob"] = test_prob[:, 0]
test_out["news_impact_medium_prob"] = test_prob[:, 1]
test_out["news_impact_high_prob"] = test_prob[:, 2]
test_out["impact_prediction_source"] = "train_validation_fitted"

out = pd.concat(
    [train_out, val_out, test_out],
    ignore_index=True,
)

out["predicted_news_impact"] = (
    np.argmax(
        out[
            [
                "news_impact_low_prob",
                "news_impact_medium_prob",
                "news_impact_high_prob",
            ]
        ].fillna(-1).values,
        axis=1,
    )
)

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

# Entropy is useful downstream:
# high entropy = model is uncertain about impact class.
probs = out[
    [
        "news_impact_low_prob",
        "news_impact_medium_prob",
        "news_impact_high_prob",
    ]
].astype(float)

safe_probs = probs.clip(lower=1e-12)

out["news_impact_entropy"] = (
    -(safe_probs * np.log(safe_probs))
    .sum(axis=1)
)

out = out.sort_values(
    ["Date", "symbol"]
).reset_index(drop=True)


# =============================================================================
# SAVE
# =============================================================================

out.to_parquet(
    OUT_FILE,
    index=False,
    compression="snappy",
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False,
)

print(f"\nSaved predictions: {OUT_FILE}")
print(f"Saved metrics    : {METRICS_FILE}")

print("\nPrediction source:")
print(
    out["impact_prediction_source"]
    .value_counts()
    .to_string()
)

print("\nPredicted impact distribution:")
print(
    out["predicted_news_impact_label"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\n" + "=" * 110)
print("STATUS: PASS — NEWS IMPACT MODEL COMPLETE")
print("=" * 110)
