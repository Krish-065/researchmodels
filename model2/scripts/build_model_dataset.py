from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.home() / "24DIT065" / "model2"

STOCK = (
    ROOT
    / "03_features"
    / "market"
    / "stock_features_daily.parquet"
)

RELATED = (
    ROOT
    / "03_features"
    / "related"
    / "related_asset_features_daily.parquet"
)

OUT = (
    ROOT
    / "04_model"
    / "datasets"
)

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 100)
print("MODEL 2 — BUILD MODEL DATASET")
print("=" * 100)

print("Loading stock features...")
stock = pd.read_parquet(STOCK)

print("Loading related features...")
related = pd.read_parquet(RELATED)

stock["Date"] = pd.to_datetime(stock["Date"])
related["Date"] = pd.to_datetime(related["Date"])

stock = stock.sort_values(["symbol", "Date"]).reset_index(drop=True)
related = related.sort_values("Date").reset_index(drop=True)

print("Stock rows   :", len(stock))
print("Related rows :", len(related))


# ============================================================
# TARGETS
# ============================================================

print()
print("Creating forward targets...")

stock["target_return_1d"] = (
    stock.groupby("symbol")["Close"]
    .shift(-1)
    / stock["Close"]
    - 1
)

stock["target_return_3d"] = (
    stock.groupby("symbol")["Close"]
    .shift(-3)
    / stock["Close"]
    - 1
)

stock["target_return_5d"] = (
    stock.groupby("symbol")["Close"]
    .shift(-5)
    / stock["Close"]
    - 1
)

stock["target_direction_1d"] = (
    stock["target_return_1d"] > 0
).astype("float")

stock["target_direction_5d"] = (
    stock["target_return_5d"] > 0
).astype("float")


# ============================================================
# RELATED-ASSET ALIGNMENT
#
# IMPORTANT:
# For every Indian stock date, use the latest AVAILABLE
# related-asset observation at or BEFORE that date.
#
# This avoids introducing NaNs simply because another market
# was closed on the Indian trading date.
#
# International assets are additionally shifted by one
# observation so their information is conservative with
# respect to the Indian trading session.
# ============================================================

print()
print("Preparing leakage-safe related features...")


# Indian-market assets can be aligned as-of the same date.
same_day_assets = [
    "nifty50",
    "indiavix",
    "usdinr",
]


# International / non-Indian assets.
# Shift their complete feature set by one available observation
# before the as-of merge.
lagged_assets = [
    "sp500",
    "nasdaq",
    "dow",
    "nikkei",
    "hangseng",
    "gold",
    "crude",
    "btc",
    "eth",
]


def prepare_related_columns(prefix, frame):

    cols = [
        c
        for c in frame.columns
        if c.startswith(prefix + "_")
    ]

    return cols


# Start with Date only.
related_aligned = related[["Date"]].copy()


# ------------------------------------------------------------
# Same-day Indian assets
# ------------------------------------------------------------

for prefix in same_day_assets:

    cols = prepare_related_columns(prefix, related)

    if not cols:
        print("WARNING: no columns found for", prefix)
        continue

    temp = related[["Date"] + cols].copy()

    temp = (
        temp
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
    )

    related_aligned = pd.merge_asof(
        related_aligned.sort_values("Date"),
        temp.sort_values("Date"),
        on="Date",
        direction="backward",
    )


# ------------------------------------------------------------
# Lagged international assets
# ------------------------------------------------------------

for prefix in lagged_assets:

    cols = prepare_related_columns(prefix, related)

    if not cols:
        print("WARNING: no columns found for", prefix)
        continue

    temp = related[["Date"] + cols].copy()

    temp = (
        temp
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
    )

    # Shift observation availability by one row.
    # At Indian date t, we use the previous available
    # international-market observation.
    temp[cols] = temp[cols].shift(1)

    related_aligned = pd.merge_asof(
        related_aligned.sort_values("Date"),
        temp.sort_values("Date"),
        on="Date",
        direction="backward",
    )


# ============================================================
# MERGE STOCK + RELATED
# ============================================================

print()
print("Merging stock and related features...")

model = stock.merge(
    related_aligned,
    on="Date",
    how="left",
    validate="many_to_one",
)

model = (
    model
    .sort_values(["Date", "symbol"])
    .reset_index(drop=True)
)


# ============================================================
# REMOVE ROWS WITHOUT TARGET
# ============================================================

before = len(model)

model = model.dropna(
    subset=[
        "target_return_1d",
        "target_return_5d",
    ]
).copy()

after = len(model)

print()
print("Rows before target filtering:", before)
print("Rows after target filtering :", after)
print("Rows removed                :", before - after)


# ============================================================
# DATA TYPE / SANITY CLEANUP
# ============================================================

model["symbol"] = model["symbol"].astype(str)

model = model.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# SAVE
# ============================================================

output = OUT / "model_dataset_daily.parquet"

model.to_parquet(
    output,
    index=False
)


# ============================================================
# AUDIT
# ============================================================

print()
print("=" * 100)
print("MODEL DATASET CREATED")
print("=" * 100)

print("Rows       :", len(model))
print("Columns    :", len(model.columns))
print("Stocks     :", model["symbol"].nunique())
print("Date start :", model["Date"].min())
print("Date end   :", model["Date"].max())

print()
print("Duplicate Date+Symbol:",
      model.duplicated(["Date", "symbol"]).sum())

print()
print("Targets:")
print(
    model[
        [
            "target_return_1d",
            "target_return_3d",
            "target_return_5d",
            "target_direction_1d",
            "target_direction_5d",
        ]
    ].describe()
)

print()
print("Target direction balance:")

print(
    model["target_direction_1d"]
    .value_counts(normalize=True)
    .sort_index()
)

print()
print("Missing values — top 25:")

print(
    model.isna()
    .sum()
    .sort_values(ascending=False)
    .head(25)
)

print()
print("Saved:")
print(output)

print()
print("=" * 100)

assert model["symbol"].nunique() == 49

assert (
    model.duplicated(
        ["Date", "symbol"]
    ).sum()
    == 0
)

assert model["target_return_1d"].notna().all()

assert model["target_return_5d"].notna().all()

assert len(model) > 80000

print("STATUS: PASS — MODEL DATASET READY")
print("=" * 100)
