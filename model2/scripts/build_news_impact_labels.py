from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("/home/administrator/24DIT065/model2")

NEWS_FILE = BASE / "03_features/news/news_daily_stock_features.parquet"
MARKET_FILE = BASE / "02_interim/yahoo/master/master_stocks_daily.parquet"

OUT_DIR = BASE / "04_model/datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "news_impact_labels.parquet"


print("=" * 100)
print("MODEL 2 — NEWS IMPACT LABEL ENGINE")
print("=" * 100)

# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------

news = pd.read_parquet(NEWS_FILE)
market = pd.read_parquet(MARKET_FILE)

news["Date"] = pd.to_datetime(news["Date"])
market["Date"] = pd.to_datetime(market["Date"])

print(f"News rows   : {len(news):,}")
print(f"Market rows : {len(market):,}")

# ---------------------------------------------------------------------
# CHECK REQUIRED MARKET COLUMNS
# ---------------------------------------------------------------------

required = ["Date", "symbol", "Open", "High", "Low", "Close", "Volume"]

missing = [c for c in required if c not in market.columns]

if missing:
    raise RuntimeError(f"Missing required market columns: {missing}")

market = market.sort_values(["symbol", "Date"]).reset_index(drop=True)

# ---------------------------------------------------------------------
# CONSTRUCT FUTURE MARKET REACTION
# ---------------------------------------------------------------------
#
# For each stock:
#
#   return_next_1d = Close(t+1) / Close(t) - 1
#
#   volume_ratio_next_1d =
#       Volume(t+1) / rolling median Volume of previous 20 sessions
#
#   volatility_20d =
#       trailing std of daily returns using information available up to t
#
# Then normalize next-day return by trailing volatility.
#
# IMPORTANT:
# These FUTURE values are LABEL INFORMATION only.
# They must never become predictor features.
# ---------------------------------------------------------------------

market["return_1d"] = (
    market.groupby("symbol")["Close"]
    .pct_change()
)

# Previous 20-session volume baseline
market["volume_median_20"] = (
    market.groupby("symbol")["Volume"]
    .transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).median()
    )
)

market["volatility_20d"] = (
    market.groupby("symbol")["return_1d"]
    .transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).std()
    )
)

# Next trading session values
market["next_close"] = (
    market.groupby("symbol")["Close"].shift(-1)
)

market["next_volume"] = (
    market.groupby("symbol")["Volume"].shift(-1)
)

market["next_return"] = (
    market["next_close"] / market["Close"] - 1.0
)

market["next_volume_ratio"] = (
    market["next_volume"] / market["volume_median_20"]
)

market["next_abs_return"] = market["next_return"].abs()

# Volatility-normalized return shock
market["next_return_z"] = (
    market["next_return"] /
    market["volatility_20d"].replace(0, np.nan)
)

market["next_abs_return_z"] = market["next_return_z"].abs()

# ---------------------------------------------------------------------
# COMBINED REALIZED IMPACT SCORE
# ---------------------------------------------------------------------
#
# Return shock is the primary signal.
# Volume shock is secondary evidence that the market actually reacted.
#
# We use log volume shock so extreme volume does not dominate.
# ---------------------------------------------------------------------

market["volume_shock"] = np.log1p(
    market["next_volume_ratio"].clip(lower=0)
)

# Clip extreme values for robustness
return_component = market["next_abs_return_z"].clip(upper=10)
volume_component = market["volume_shock"].clip(-2, 5)

market["realized_impact_score"] = (
    0.75 * return_component +
    0.25 * volume_component
)

# ---------------------------------------------------------------------
# LABEL NEWS DAYS
# ---------------------------------------------------------------------

label_cols = [
    "Date",
    "symbol",
    "Close",
    "Volume",
    "return_1d",
    "volatility_20d",
    "next_return",
    "next_abs_return",
    "next_volume_ratio",
    "next_abs_return_z",
    "realized_impact_score",
]

labels = market[label_cols].copy()

# Only dates/stocks for which news exists
labels = labels.merge(
    news[["Date", "symbol", "news_count", "news_mean_sentiment",
          "news_intensity", "news_sentiment_pressure"]],
    on=["Date", "symbol"],
    how="inner",
)

# ---------------------------------------------------------------------
# CREATE IMPACT CLASSES
# ---------------------------------------------------------------------
#
# We use cross-sectional/global empirical quantiles rather than arbitrary
# percentage thresholds. This makes the classes better suited to this
# heterogeneous 49-stock universe.
#
# LOW    = bottom 50%
# MEDIUM = 50-80%
# HIGH   = top 20%
# ---------------------------------------------------------------------

valid = labels["realized_impact_score"].dropna()

if len(valid) < 100:
    raise RuntimeError(
        f"Too few valid impact observations: {len(valid)}"
    )

q50 = valid.quantile(0.50)
q80 = valid.quantile(0.80)

labels["impact_class"] = np.select(
    [
        labels["realized_impact_score"] < q50,
        labels["realized_impact_score"] < q80,
    ],
    [
        "LOW",
        "MEDIUM",
    ],
    default="HIGH",
)

labels["impact_class_id"] = labels["impact_class"].map(
    {
        "LOW": 0,
        "MEDIUM": 1,
        "HIGH": 2,
    }
)

# ---------------------------------------------------------------------
# CLEAN
# ---------------------------------------------------------------------

labels = labels.replace([np.inf, -np.inf], np.nan)

labels = labels.dropna(
    subset=[
        "realized_impact_score",
        "next_return",
        "next_abs_return",
        "next_volume_ratio",
        "next_abs_return_z",
    ]
)

labels = labels.sort_values(
    ["Date", "symbol"]
).reset_index(drop=True)

# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

labels.to_parquet(
    OUT_FILE,
    index=False,
    compression="snappy",
)

# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

print("\n" + "=" * 100)
print("RESULT")
print("=" * 100)

print(f"Rows              : {len(labels):,}")
print(f"Stocks             : {labels['symbol'].nunique()}")
print(f"Date range         : {labels['Date'].min()} -> {labels['Date'].max()}")

print("\nImpact thresholds:")
print(f"LOW/MEDIUM boundary : {q50:.6f}")
print(f"MEDIUM/HIGH boundary: {q80:.6f}")

print("\nImpact distribution:")
print(
    labels["impact_class"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nImpact distribution (%):")
print(
    (
        labels["impact_class"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
        .astype(str)
        + "%"
    ).to_string()
)

print("\nRealized next-day return:")
print(labels["next_return"].describe().to_string())

print("\nNext-day absolute return:")
print(labels["next_abs_return"].describe().to_string())

print("\nNext-day volume ratio:")
print(labels["next_volume_ratio"].describe().to_string())

print("\nRealized impact score:")
print(labels["realized_impact_score"].describe().to_string())

print(f"\nSaved: {OUT_FILE}")

print("\n" + "=" * 100)
print("STATUS: PASS — REALIZED NEWS IMPACT LABELS READY")
print("=" * 100)
