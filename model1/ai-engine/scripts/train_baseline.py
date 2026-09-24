from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)


INPUT = Path("/workspace/data/features/nse_features.parquet")
MODEL_DIR = Path("/workspace/data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lightgbm_baseline.txt"


FEATURES = [
    "symbol_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "sma_5",
    "sma_20",
    "sma_50",
    "sma_200",
    "close_sma5_ratio",
    "close_sma20_ratio",
    "close_sma50_ratio",
    "close_sma200_ratio",
    "ema_12",
    "ema_26",
    "macd",
    "macd_signal",
    "rsi_14",
    "volatility_20d",
    "volatility_60d",
    "high_low_range",
    "open_close_return",
    "volume_change",
    "volume_zscore",
]


def main() -> None:
    print("=" * 70)
    print("LIGHTGBM BASELINE")
    print("=" * 70)

    df = pd.read_parquet(INPUT)

    # Encode stock identity without using the target.
    # Same symbol always receives the same integer.
    symbols = {s: i for i, s in enumerate(sorted(df["symbol"].unique()))}
    df["symbol_id"] = df["symbol"].map(symbols).astype("int16")

    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    # IMPORTANT:
    # Chronological split. No random train/test split.
    train = df[df["timestamp"] < "2024-01-01"].copy()
    valid = df[
        (df["timestamp"] >= "2024-01-01")
        & (df["timestamp"] < "2025-01-01")
    ].copy()
    test = df[df["timestamp"] >= "2025-01-01"].copy()

    print("Total :", f"{len(df):,}")
    print("Train :", f"{len(train):,}")
    print("Valid :", f"{len(valid):,}")
    print("Test  :", f"{len(test):,}")

    print()
    print("Date ranges:")
    for name, part in [
        ("TRAIN", train),
        ("VALID", valid),
        ("TEST ", test),
    ]:
        print(
            f"{name}: "
            f"{part.timestamp.min()} -> {part.timestamp.max()}"
        )

    X_train = train[FEATURES]
    y_train = train["target_up_5d"].astype(int)

    X_valid = valid[FEATURES]
    y_valid = valid["target_up_5d"].astype(int)

    X_test = test[FEATURES]
    y_test = test["target_up_5d"].astype(int)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    print()
    print("Training...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(75, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    valid_prob = model.predict_proba(X_valid)[:, 1]
    valid_pred = (valid_prob >= 0.5).astype(int)

    print("Accuracy          :", accuracy_score(y_valid, valid_pred))
    print("Balanced accuracy :", balanced_accuracy_score(y_valid, valid_pred))
    print("ROC-AUC           :", roc_auc_score(y_valid, valid_prob))

    print()
    print(classification_report(y_valid, valid_pred))

    print("=" * 70)
    print("TEST")
    print("=" * 70)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    print("Accuracy          :", accuracy_score(y_test, test_pred))
    print("Balanced accuracy :", balanced_accuracy_score(y_test, test_pred))
    print("ROC-AUC           :", roc_auc_score(y_test, test_prob))

    print()
    print(classification_report(y_test, test_pred))

    model.booster_.save_model(str(MODEL_PATH))

    print()
    print("MODEL SAVED:", MODEL_PATH)

    importance = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
    )

    print()
    print("TOP FEATURES:")
    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
