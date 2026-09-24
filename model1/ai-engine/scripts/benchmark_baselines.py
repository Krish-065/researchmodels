from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


INPUT = Path("/workspace/data/features/nse_features.parquet")


# Same feature set for every model.
FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "close_sma5_ratio",
    "close_sma20_ratio",
    "close_sma50_ratio",
    "close_sma200_ratio",
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


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "PR-AUC": average_precision_score(y_test, prob),
        "Brier": brier_score_loss(y_test, prob),
    }

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    print(f"Accuracy          : {metrics['Accuracy']:.6f}")
    print(f"Balanced accuracy : {metrics['Balanced Accuracy']:.6f}")
    print(f"ROC-AUC           : {metrics['ROC-AUC']:.6f}")
    print(f"PR-AUC            : {metrics['PR-AUC']:.6f}")
    print(f"Brier score       : {metrics['Brier']:.6f}")

    return metrics


def main():
    print("=" * 70)
    print("STANDARDIZED MODEL BENCHMARK")
    print("=" * 70)

    df = pd.read_parquet(INPUT)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values(
        ["timestamp", "symbol"]
    ).reset_index(drop=True)

    # Strict chronological split.
    train = df[df["timestamp"] < "2024-01-01"].copy()

    valid = df[
        (df["timestamp"] >= "2024-01-01")
        & (df["timestamp"] < "2025-01-01")
    ].copy()

    test = df[df["timestamp"] >= "2025-01-01"].copy()

    X_train = (
        train[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
    )

    X_test = (
        test[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
    )

    y_train = train["target_up_5d"].astype(int)
    y_test = test["target_up_5d"].astype(int)

    print()
    print("DATA")
    print("-" * 70)
    print(f"Total : {len(df):,}")
    print(f"Train : {len(train):,}")
    print(f"Valid : {len(valid):,}")
    print(f"Test  : {len(test):,}")
    print(f"Features: {len(FEATURES)}")

    print()
    print("DATE RANGES")
    print("-" * 70)
    print(
        "TRAIN:",
        train["timestamp"].min(),
        "->",
        train["timestamp"].max(),
    )
    print(
        "VALID:",
        valid["timestamp"].min(),
        "->",
        valid["timestamp"].max(),
    )
    print(
        "TEST :",
        test["timestamp"].min(),
        "->",
        test["timestamp"].max(),
    )

    print()
    print("TARGET")
    print("-" * 70)
    print(f"Train target rate: {y_train.mean():.6f}")
    print(f"Test target rate : {y_test.mean():.6f}")

    # ---------------------------------------------------------------
    # MODELS
    # ---------------------------------------------------------------

    models = [
        (
            "DUMMY / PRIOR",
            DummyClassifier(
                strategy="prior"
            ),
        ),

        (
            "LOGISTIC REGRESSION",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            C=1.0,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),

        (
            "RANDOM FOREST",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=8,
                            min_samples_leaf=50,
                            max_features="sqrt",
                            n_jobs=-1,
                            random_state=42,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        ),

        (
            "XGBOOST",
            XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=20,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            ),
        ),

        (
            "LIGHTGBM",
            LGBMClassifier(
                n_estimators=500,
                num_leaves=31,
                max_depth=-1,
                learning_rate=0.03,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="binary",
                verbosity=-1,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]

    results = []

    print()
    print("=" * 70)
    print("MODEL RESULTS")
    print("=" * 70)

    for name, model in models:
        result = evaluate(
            name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=" * 70)
    print("ROC-AUC RANKING")
    print("=" * 70)

    ranking = results_df.sort_values(
        "ROC-AUC",
        ascending=False,
    )

    for _, row in ranking.iterrows():
        print(
            f"{row['Model']:<25} "
            f"ROC-AUC={row['ROC-AUC']:.6f}"
        )


if __name__ == "__main__":
    main()
