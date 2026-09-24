from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("/home/administrator/24DIT065/model2")

MODEL_BASE = BASE / "04_model/datasets"

MARKET = BASE / "03_features/market/stock_features_daily.parquet"
RELATED = BASE / "03_features/related/related_asset_features_daily.parquet"
NEWS = BASE / "03_features/news/news_daily_stock_features.parquet"
IMPACT = MODEL_BASE / "news_impact_predictions_v2.parquet"

OUT = MODEL_BASE / "multimodal_dataset_daily.parquet"

print("=" * 110)
print("MODEL 2 — MULTIMODAL DAILY DATASET")
print("=" * 110)

# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------

market = pd.read_parquet(MARKET)
related = pd.read_parquet(RELATED)
news = pd.read_parquet(NEWS)
impact = pd.read_parquet(IMPACT)

for df in [market, related, news, impact]:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

print("\nInput sizes:")
print("Market :", market.shape)
print("Related:", related.shape)
print("News   :", news.shape)
print("Impact :", impact.shape)

# ---------------------------------------------------------------------
# MARKET BASE
# ---------------------------------------------------------------------

market = market.sort_values(
    ["Date", "symbol"]
).drop_duplicates(
    ["Date", "symbol"]
)

# These are the target columns already used in Model 2.
# Create them here from actual future prices to keep the multimodal
# dataset self-contained.
required = ["Date", "symbol", "Close"]

missing = [c for c in required if c not in market.columns]

if missing:
    raise RuntimeError(
        f"Market dataset missing: {missing}"
    )

market["target_return_1d"] = (
    market.groupby("symbol")["Close"]
    .shift(-1) / market["Close"] - 1.0
)

market["target_return_3d"] = (
    market.groupby("symbol")["Close"]
    .shift(-3) / market["Close"] - 1.0
)

market["target_return_5d"] = (
    market.groupby("symbol")["Close"]
    .shift(-5) / market["Close"] - 1.0
)

market["target_direction_1d"] = (
    market["target_return_1d"] > 0
).astype("float")

market["target_direction_5d"] = (
    market["target_return_5d"] > 0
).astype("float")

# ---------------------------------------------------------------------
# REMOVE NON-FEATURE / RAW FIELDS FROM MARKET
# ---------------------------------------------------------------------

drop_market = {
    "Date",
    "symbol",
    "Close",
    "Open",
    "High",
    "Low",
    "Adj Close",
    "Volume",
    "target_return_1d",
    "target_return_3d",
    "target_return_5d",
    "target_direction_1d",
    "target_direction_5d",
}

market_features = [
    c for c in market.columns
    if c not in drop_market
    and pd.api.types.is_numeric_dtype(market[c])
]

market = market[
    ["Date", "symbol"] +
    market_features +
    [
        "target_return_1d",
        "target_return_3d",
        "target_return_5d",
        "target_direction_1d",
        "target_direction_5d",
    ]
]

# ---------------------------------------------------------------------
# RELATED ASSETS
# ---------------------------------------------------------------------

related = related.sort_values("Date")

# Related file should be Date-indexed rather than Date × symbol.
# Remove duplicate dates if necessary.
related = related.drop_duplicates("Date")

related_features = [
    c for c in related.columns
    if c != "Date"
    and pd.api.types.is_numeric_dtype(related[c])
]

related = related[
    ["Date"] + related_features
]

# ---------------------------------------------------------------------
# NEWS
# ---------------------------------------------------------------------

news_features = [
    c for c in news.columns
    if c not in ["Date", "symbol"]
    and pd.api.types.is_numeric_dtype(news[c])
]

news = news[
    ["Date", "symbol"] + news_features
]

news = news.drop_duplicates(
    ["Date", "symbol"]
)

# Prefix to avoid naming collisions.
news = news.rename(
    columns={
        c: f"news_{c}"
        for c in news_features
    }
)

# ---------------------------------------------------------------------
# IMPACT
# ---------------------------------------------------------------------

impact_cols = [
    "Date",
    "symbol",
    "predicted_news_impact_score",
    "news_impact_low_prob",
    "news_impact_medium_prob",
    "news_impact_high_prob",
    "news_impact_entropy",
]

impact_cols = [
    c for c in impact_cols
    if c in impact.columns
]

impact = impact[
    impact_cols
].drop_duplicates(
    ["Date", "symbol"]
)

# Rename explicitly.
impact = impact.rename(
    columns={
        "predicted_news_impact_score":
            "news_impact_score",
        "news_impact_low_prob":
            "news_impact_low_prob_v2",
        "news_impact_medium_prob":
            "news_impact_medium_prob_v2",
        "news_impact_high_prob":
            "news_impact_high_prob_v2",
        "news_impact_entropy":
            "news_impact_entropy_v2",
    }
)

# ---------------------------------------------------------------------
# MERGE
# ---------------------------------------------------------------------

print("\nMerging market + related...")

df = market.merge(
    related,
    on="Date",
    how="left",
)

print("After related:", df.shape)

df = df.merge(
    news,
    on=["Date", "symbol"],
    how="left",
)

print("After news:", df.shape)

df = df.merge(
    impact,
    on=["Date", "symbol"],
    how="left",
)

print("After impact:", df.shape)

# ---------------------------------------------------------------------
# NEWS DEFAULTS
# ---------------------------------------------------------------------
#
# No news on a day is different from unknown data:
# for the daily model, absence of matched news becomes zero news activity.
#
# Impact score remains 0 when no news is present.
# ---------------------------------------------------------------------

news_numeric = [
    c for c in df.columns
    if c.startswith("news_")
]

for c in news_numeric:
    if c in df.columns:
        df[c] = df[c].fillna(0.0)

# ---------------------------------------------------------------------
# SORT
# ---------------------------------------------------------------------

df = (
    df.sort_values(["Date", "symbol"])
      .reset_index(drop=True)
)

# ---------------------------------------------------------------------
# CHECK DUPLICATES
# ---------------------------------------------------------------------

dup = df.duplicated(
    ["Date", "symbol"]
).sum()

if dup:
    raise RuntimeError(
        f"Duplicate Date+symbol rows detected: {dup}"
    )

# ---------------------------------------------------------------------
# FEATURE / TARGET SEPARATION
# ---------------------------------------------------------------------

target_cols = [
    "target_return_1d",
    "target_return_3d",
    "target_return_5d",
    "target_direction_1d",
    "target_direction_5d",
]

feature_cols = [
    c for c in df.columns
    if c not in ["Date", "symbol"] + target_cols
    and pd.api.types.is_numeric_dtype(df[c])
]

# Raw target rows need to exist for the final modeling period.
before = len(df)

df = df.dropna(
    subset=[
        "target_return_1d",
        "target_return_5d",
    ]
).reset_index(drop=True)

print(
    f"\nDropped {before - len(df):,} rows "
    f"without future targets."
)

# ---------------------------------------------------------------------
# DATA QUALITY
# ---------------------------------------------------------------------

print("\nFinal dataset:")
print("Rows   :", f"{len(df):,}")
print("Stocks :", df["symbol"].nunique())
print("Dates  :", df["Date"].min(), "->", df["Date"].max())
print("Features:", len(feature_cols))

print("\nNews columns:")
for c in news_numeric:
    print(" ", c)

print("\nImpact columns:")
for c in [
    "news_impact_score",
    "news_impact_low_prob_v2",
    "news_impact_medium_prob_v2",
    "news_impact_high_prob_v2",
    "news_impact_entropy_v2",
]:
    if c in df.columns:
        print(" ", c)

# ---------------------------------------------------------------------
# FINITE CHECK
# ---------------------------------------------------------------------

numeric = df[feature_cols + target_cols]

finite_bad = np.isinf(
    numeric.to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
).sum()

print("\nInfinite numeric values:", finite_bad)

# NaNs are expected in technical warmups and impact predictions.
# Report rather than blindly filling everything here.
nan_counts = (
    df[feature_cols]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print("\nTop NaN features:")
print(
    nan_counts.head(20).to_string()
)

# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

df.to_parquet(
    OUT,
    index=False,
    compression="snappy",
)

# Feature manifest
manifest = pd.DataFrame(
    {
        "feature": feature_cols,
        "dtype": [
            str(df[c].dtype)
            for c in feature_cols
        ],
    }
)

manifest.to_csv(
    MODEL_BASE / "multimodal_feature_manifest.csv",
    index=False,
)

print("\nSaved:")
print(OUT)

print(
    MODEL_BASE /
    "multimodal_feature_manifest.csv"
)

print("\n" + "=" * 110)
print("STATUS: PASS — MULTIMODAL DATASET CREATED")
print("=" * 110)
