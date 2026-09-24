from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path.home() / "24DIT065" / "model2"

DATA = ROOT / "04_model" / "datasets" / "model_dataset_daily.parquet"
FEATURES = ROOT / "04_model" / "datasets" / "feature_list.csv"

OUT = ROOT / "04_model" / "results"
MODELS = ROOT / "04_model" / "models"

OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

TARGET = "target_return_1d"

TRAIN_END = pd.Timestamp("2023-12-31")
VALID_END = pd.Timestamp("2025-12-31")

RANDOM_STATE = 42


# ============================================================
# LOAD
# ============================================================

print("=" * 100)
print("MODEL 2 — HISTOGRAM GRADIENT BOOSTING BASELINE")
print("=" * 100)

df = pd.read_parquet(DATA)

feature_list = pd.read_csv(FEATURES)["feature"].tolist()

print("Dataset shape:", df.shape)
print("Features     :", len(feature_list))
print("Target       :", TARGET)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

train = df[df["Date"] <= TRAIN_END].copy()

valid = df[
    (df["Date"] > TRAIN_END)
    & (df["Date"] <= VALID_END)
].copy()

test = df[df["Date"] > VALID_END].copy()


print()
print("TRAIN :", train.shape)
print("VALID :", valid.shape)
print("TEST  :", test.shape)


# ============================================================
# MATRICES
# ============================================================

X_train = train[feature_list].astype(np.float32)
X_valid = valid[feature_list].astype(np.float32)
X_test = test[feature_list].astype(np.float32)

y_train = train[TARGET].astype(np.float32)
y_valid = valid[TARGET].astype(np.float32)
y_test = test[TARGET].astype(np.float32)


# ============================================================
# MODEL
# ============================================================

print()
print("Training HistGradientBoostingRegressor...")

model = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=30,
    random_state=RANDOM_STATE,
    verbose=1,
)


start = time.time()

model.fit(
    X_train,
    y_train,
)

elapsed = time.time() - start

print()
print("Training time:", round(elapsed, 2), "seconds")
print("Iterations   :", model.n_iter_)


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("Generating predictions...")

pred_train = model.predict(X_train)
pred_valid = model.predict(X_valid)
pred_test = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

def regression_metrics(y, pred):

    mae = mean_absolute_error(y, pred)

    rmse = np.sqrt(
        mean_squared_error(y, pred)
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
    }


def cross_sectional_ic(frame, predictions):

    temp = frame[
        ["Date", "symbol", TARGET]
    ].copy()

    temp["prediction"] = predictions

    daily_ic = []

    for _, g in temp.groupby("Date"):

        if len(g) < 3:
            continue

        if g[TARGET].nunique() < 2:
            continue

        if g["prediction"].nunique() < 2:
            continue

        ic = g["prediction"].corr(
            g[TARGET],
            method="spearman",
        )

        if pd.notna(ic):
            daily_ic.append(ic)

    daily_ic = np.asarray(daily_ic)

    if len(daily_ic) == 0:
        return {
            "Mean_IC": np.nan,
            "IC_STD": np.nan,
            "ICIR": np.nan,
        }

    mean_ic = daily_ic.mean()
    std_ic = daily_ic.std(ddof=1)

    if std_ic == 0:
        icir = np.nan
    else:
        icir = mean_ic / std_ic * np.sqrt(252)

    return {
        "Mean_IC": float(mean_ic),
        "IC_STD": float(std_ic),
        "ICIR": float(icir),
    }


def evaluate(name, frame, y, pred):

    result = {}

    result.update(
        regression_metrics(y, pred)
    )

    result.update(
        cross_sectional_ic(
            frame,
            pred,
        )
    )

    print()
    print(name)
    print("-" * 80)

    for k, v in result.items():
        print(f"{k:12s}: {v:.8f}")

    return result


results = {}

results["TRAIN"] = evaluate(
    "TRAIN",
    train,
    y_train,
    pred_train,
)

results["VALID"] = evaluate(
    "VALID",
    valid,
    y_valid,
    pred_valid,
)

results["TEST"] = evaluate(
    "TEST",
    test,
    y_test,
    pred_test,
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = pd.concat(
    [
        pd.DataFrame({
            "Date": train["Date"].values,
            "symbol": train["symbol"].values,
            "split": "train",
            "actual_return_1d": y_train.values,
            "prediction": pred_train,
        }),

        pd.DataFrame({
            "Date": valid["Date"].values,
            "symbol": valid["symbol"].values,
            "split": "valid",
            "actual_return_1d": y_valid.values,
            "prediction": pred_valid,
        }),

        pd.DataFrame({
            "Date": test["Date"].values,
            "symbol": test["symbol"].values,
            "split": "test",
            "actual_return_1d": y_test.values,
            "prediction": pred_test,
        }),
    ],
    ignore_index=True,
)

prediction_path = OUT / "hgb_baseline_predictions.parquet"

predictions.to_parquet(
    prediction_path,
    index=False,
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame(results).T
metrics.index.name = "split"

metrics_path = OUT / "hgb_baseline_metrics.csv"

metrics.to_csv(metrics_path)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODELS / "hgb_baseline.joblib"

joblib.dump(
    model,
    model_path,
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if hasattr(model, "feature_importances_"):

    importance = pd.DataFrame({
        "feature": feature_list,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        OUT / "hgb_baseline_feature_importance.csv",
        index=False,
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 100)
print("HGB BASELINE COMPLETE")
print("=" * 100)

print("Validation IC :", results["VALID"]["Mean_IC"])
print("Validation ICIR:", results["VALID"]["ICIR"])

print("Test IC       :", results["TEST"]["Mean_IC"])
print("Test ICIR     :", results["TEST"]["ICIR"])

print()
print("Saved predictions:")
print(prediction_path)

print()
print("Saved metrics:")
print(metrics_path)

print()
print("Saved model:")
print(model_path)

print("=" * 100)
