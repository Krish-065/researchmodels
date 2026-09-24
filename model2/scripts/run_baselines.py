import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path.home() / "24DIT065" / "model2"

DATA = ROOT / "04_model" / "datasets" / "model_dataset_daily.parquet"
SPLITS = ROOT / "04_model" / "splits"

print("=" * 100)
print("MODEL 2 — BASELINE EXPERIMENT")
print("=" * 100)

df = pd.read_parquet(DATA)
df["Date"] = pd.to_datetime(df["Date"])

train = df[
    (df["Date"] >= "2019-01-01") &
    (df["Date"] <= "2023-12-31")
].copy()

valid = df[
    (df["Date"] >= "2024-01-01") &
    (df["Date"] <= "2025-12-31")
].copy()

test = df[
    (df["Date"] >= "2026-01-01") &
    (df["Date"] <= "2026-12-31")
].copy()


def evaluate(data, prediction_col, target_col):

    y = data[target_col].to_numpy()
    p = data[prediction_col].to_numpy()

    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))

    daily_ic = []

    for date, g in data.groupby("Date"):

        if len(g) < 5:
            continue

        a = g[target_col]
        b = g[prediction_col]

        ic = a.corr(b, method="spearman")

        if pd.notna(ic):
            daily_ic.append(ic)

    daily_ic = np.asarray(daily_ic)

    mean_ic = np.mean(daily_ic)

    ic_std = np.std(
        daily_ic,
        ddof=1
    )

    icir = (
        mean_ic / ic_std
        if ic_std > 0
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Mean_IC": mean_ic,
        "IC_STD": ic_std,
        "ICIR": icir,
    }


# =====================================================================
# BASELINE 1 — ZERO RETURN
# =====================================================================

for name, data in [
    ("TRAIN", train),
    ("VALID", valid),
    ("TEST", test),
]:

    data = data.copy()

    data["prediction"] = 0.0

    metrics = evaluate(
        data,
        "prediction",
        "target_return_1d"
    )

    print()
    print(name)
    print(metrics)


# =====================================================================
# BASELINE 2 — TRAINING MEAN
# =====================================================================

train_mean = train["target_return_1d"].mean()

print()
print("Training mean return:", train_mean)

for name, data in [
    ("TRAIN", train),
    ("VALID", valid),
    ("TEST", test),
]:

    data = data.copy()

    data["prediction"] = train_mean

    metrics = evaluate(
        data,
        "prediction",
        "target_return_1d"
    )

    print()
    print(name)
    print(metrics)


# =====================================================================
# CROSS-SECTIONAL RANK BASELINE
#
# Predict each stock's return using its historical average return
# estimated ONLY from the training period.
# =====================================================================

stock_mean = (
    train
    .groupby("symbol")["target_return_1d"]
    .mean()
)

print()
print("Number of stock historical means:", len(stock_mean))

for name, data in [
    ("TRAIN", train),
    ("VALID", valid),
    ("TEST", test),
]:

    data = data.copy()

    data["prediction"] = (
        data["symbol"]
        .map(stock_mean)
        .fillna(train_mean)
    )

    metrics = evaluate(
        data,
        "prediction",
        "target_return_1d"
    )

    print()
    print(name)
    print(metrics)


# =====================================================================
# DAILY CROSS-SECTIONAL MEAN BASELINE
# =====================================================================

for name, data in [
    ("TRAIN", train),
    ("VALID", valid),
    ("TEST", test),
]:

    data = data.copy()

    data["prediction"] = (
        data
        .groupby("Date")["target_return_1d"]
        .transform("mean")
    )

    metrics = evaluate(
        data,
        "prediction",
        "target_return_1d"
    )

    print()
    print(name, "DAILY-MEAN")

    print(metrics)


print()
print("=" * 100)
print("BASELINE EXPERIMENT COMPLETE")
print("=" * 100)
