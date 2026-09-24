from pathlib import Path
import ast
import numpy as np
import pandas as pd

BASE = Path("/home/administrator/24DIT065/model2")

NEWS_FILE = BASE / "03_features/news/news_daily_stock_features.parquet"
MARKET_FILE = BASE / "02_interim/yahoo/master/master_stocks_daily.parquet"
NIFTY_FILE = BASE / "01_raw/yahoo/historical/INDEX_NSEI_daily.parquet"

OUT_DIR = BASE / "04_model/datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "news_impact_labels_v2.parquet"


print("=" * 110)
print("MODEL 2 — NEWS IMPACT V2")
print("Stock-normalized + NIFTY-relative + abnormal volume")
print("=" * 110)


# =============================================================================
# LOAD
# =============================================================================

news = pd.read_parquet(NEWS_FILE)
market = pd.read_parquet(MARKET_FILE)
nifty = pd.read_parquet(NIFTY_FILE)

news["Date"] = pd.to_datetime(news["Date"])
market["Date"] = pd.to_datetime(market["Date"])


# =============================================================================
# ROBUST NIFTY COLUMN NORMALIZATION
# =============================================================================

def normalize_yahoo_columns(df):
    """
    Handles:
      1. Real pandas MultiIndex columns
      2. Tuple-like strings such as:
         "('Date', '')"
         "('Close', '^NSEI')"
      3. Ordinary flat columns
    """

    if isinstance(df.columns, pd.MultiIndex):
        out = []

        for col in df.columns:
            parts = [str(x) for x in col if str(x) not in ("", "nan", "None")]

            if len(parts) == 1:
                out.append(parts[0])
            else:
                # Keep first level where possible.
                out.append(parts[0])

        df = df.copy()
        df.columns = out
        return df

    # Serialized MultiIndex / tuples
    parsed = []

    for col in df.columns:

        if isinstance(col, str):
            s = col.strip()

            if s.startswith("(") and s.endswith(")"):
                try:
                    value = ast.literal_eval(s)

                    if isinstance(value, tuple):
                        parts = [
                            str(x)
                            for x in value
                            if str(x) not in ("", "nan", "None")
                        ]

                        if len(parts) >= 1:
                            parsed.append(parts[0])
                            continue

                except Exception:
                    pass

        parsed.append(str(col))

    df = df.copy()
    df.columns = parsed

    return df


nifty = normalize_yahoo_columns(nifty)

print("\nNIFTY normalized columns:")
for c in nifty.columns:
    print(" ", c)


# =============================================================================
# FIND NIFTY DATE / CLOSE
# =============================================================================

def find_column(df, candidates):
    lower = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in lower:
            return lower[key]

    return None


nifty_date_col = find_column(
    nifty,
    [
        "Date",
        "Datetime",
        "Timestamp",
    ],
)

nifty_close_col = find_column(
    nifty,
    [
        "Close",
        "Adj Close",
        "Adj_Close",
    ],
)

if nifty_date_col is None:
    raise RuntimeError(
        "Could not identify NIFTY date column.\n"
        f"Columns: {list(nifty.columns)}"
    )

if nifty_close_col is None:
    raise RuntimeError(
        "Could not identify NIFTY close column.\n"
        f"Columns: {list(nifty.columns)}"
    )

print("\nUsing NIFTY:")
print(" Date :", nifty_date_col)
print(" Close:", nifty_close_col)


nifty = nifty[
    [nifty_date_col, nifty_close_col]
].copy()

nifty.columns = [
    "Date",
    "NIFTY_Close",
]

nifty["Date"] = pd.to_datetime(
    nifty["Date"],
    errors="coerce",
)

nifty["NIFTY_Close"] = pd.to_numeric(
    nifty["NIFTY_Close"],
    errors="coerce",
)

nifty = (
    nifty.dropna(subset=["Date", "NIFTY_Close"])
         .sort_values("Date")
         .drop_duplicates("Date")
         .reset_index(drop=True)
)

nifty["NIFTY_return_1d"] = (
    nifty["NIFTY_Close"].pct_change()
)

print(
    f"\nNIFTY rows: {len(nifty):,}"
)

print(
    f"NIFTY dates: "
    f"{nifty['Date'].min()} -> {nifty['Date'].max()}"
)


# =============================================================================
# MARKET DATA
# =============================================================================

required_market = [
    "Date",
    "symbol",
    "Close",
    "Volume",
]

missing = [
    c for c in required_market
    if c not in market.columns
]

if missing:
    raise RuntimeError(
        f"Missing market columns: {missing}"
    )

market = market[
    required_market
].copy()

market["Close"] = pd.to_numeric(
    market["Close"],
    errors="coerce",
)

market["Volume"] = pd.to_numeric(
    market["Volume"],
    errors="coerce",
)

market = (
    market.dropna(
        subset=["Date", "symbol", "Close"]
    )
    .sort_values(["symbol", "Date"])
    .reset_index(drop=True)
)

print(
    f"\nMarket rows: {len(market):,}"
)
print(
    f"Market stocks: {market['symbol'].nunique()}"
)


# =============================================================================
# DAILY STOCK RETURN
# =============================================================================

market["stock_return_1d"] = (
    market.groupby("symbol")["Close"]
    .pct_change()
)


# =============================================================================
# PRE-EVENT VOLATILITY
# =============================================================================

market["pre_vol_20d"] = (
    market.groupby("symbol")["stock_return_1d"]
    .transform(
        lambda s:
        s.shift(1)
         .rolling(20, min_periods=10)
         .std()
    )
)

market["pre_vol_60d"] = (
    market.groupby("symbol")["stock_return_1d"]
    .transform(
        lambda s:
        s.shift(1)
         .rolling(60, min_periods=30)
         .std()
    )
)


# =============================================================================
# PRE-EVENT VOLUME BASELINES
# =============================================================================

market["pre_volume_median_20"] = (
    market.groupby("symbol")["Volume"]
    .transform(
        lambda s:
        s.shift(1)
         .rolling(20, min_periods=10)
         .median()
    )
)

market["pre_volume_log"] = np.log1p(
    market["Volume"].clip(lower=0)
)

market["pre_volume_median_log_60"] = (
    market.groupby("symbol")["pre_volume_log"]
    .transform(
        lambda s:
        s.shift(1)
         .rolling(60, min_periods=30)
         .median()
    )
)

market["pre_volume_mad_60"] = (
    market.groupby("symbol")["pre_volume_log"]
    .transform(
        lambda s:
        s.shift(1)
         .rolling(60, min_periods=30)
         .apply(
             lambda x:
             np.median(
                 np.abs(
                     x - np.median(x)
                 )
             ),
             raw=True,
         )
    )
)


# =============================================================================
# MERGE NIFTY
# =============================================================================

market = market.merge(
    nifty[
        [
            "Date",
            "NIFTY_return_1d",
        ]
    ],
    on="Date",
    how="left",
)

market = market.sort_values(
    ["symbol", "Date"]
).reset_index(drop=True)


# =============================================================================
# PRE-EVENT ROLLING BETA
# =============================================================================
#
# IMPORTANT:
# all beta inputs are shifted by one session.
# Therefore beta at D only uses information <= D-1.
# =============================================================================

market["stock_x_nifty"] = (
    market["stock_return_1d"] *
    market["NIFTY_return_1d"]
)

market["stock_mean_60"] = (
    market.groupby("symbol")["stock_return_1d"]
    .transform(
        lambda s:
        s.shift(1)
         .rolling(60, min_periods=30)
         .mean()
    )
)

market["nifty_mean_60"] = (
    market.groupby("symbol")["NIFTY_return_1d"]
    .transform(
        lambda s:
        s.shift(1)
         .rolling(60, min_periods=30)
         .mean()
    )
)

market["cross_mean_60"] = (
    market.groupby("symbol")["stock_x_nifty"]
    .transform(
        lambda s:
        s.shift(1)
         .rolling(60, min_periods=30)
         .mean()
    )
)

market["nifty_sq_60"] = (
    market.groupby("symbol")["NIFTY_return_1d"]
    .transform(
        lambda s:
        s.shift(1)
         .rolling(60, min_periods=30)
         .mean()
    )
)

market["covariance_60"] = (
    market["cross_mean_60"] -
    market["stock_mean_60"] *
    market["nifty_mean_60"]
)

market["nifty_variance_60"] = (
    market.groupby("symbol")["NIFTY_return_1d"]
    .transform(
        lambda s:
        s.shift(1)
         .rolling(60, min_periods=30)
         .var()
    )
)

market["beta_60"] = (
    market["covariance_60"] /
    market["nifty_variance_60"]
        .replace(0, np.nan)
)

market["beta_60"] = (
    market["beta_60"]
    .replace([np.inf, -np.inf], np.nan)
    .clip(-5, 5)
)


# =============================================================================
# FUTURE / NEXT-TRADING-DAY REACTION
# =============================================================================

market["next_close"] = (
    market.groupby("symbol")["Close"]
    .shift(-1)
)

market["next_volume"] = (
    market.groupby("symbol")["Volume"]
    .shift(-1)
)

market["next_return"] = (
    market["next_close"] /
    market["Close"] - 1.0
)

# The next trading day's NIFTY return
market["next_nifty_return"] = (
    market.groupby("symbol")["NIFTY_return_1d"]
    .shift(-1)
)

# Market-relative next-day reaction
market["abnormal_next_return"] = (
    market["next_return"] -
    market["beta_60"] *
    market["next_nifty_return"]
)


# =============================================================================
# STOCK-SPECIFIC RETURN SHOCK
# =============================================================================

market["abnormal_return_z"] = (
    market["abnormal_next_return"] /
    market["pre_vol_60d"].replace(0, np.nan)
)


# =============================================================================
# NEXT-DAY VOLUME SHOCK
# =============================================================================

market["next_volume_ratio"] = (
    market["next_volume"] /
    market["pre_volume_median_20"]
)

next_volume_log = np.log1p(
    market["next_volume"].clip(lower=0)
)

market["volume_shock_z"] = (
    next_volume_log -
    market["pre_volume_median_log_60"]
) / (
    1.4826 *
    market["pre_volume_mad_60"]
).replace(0, np.nan)


# =============================================================================
# ROBUST TRANSFORMATION
# =============================================================================

return_component = np.tanh(
    market["abnormal_return_z"]
    .abs()
    .clip(upper=20)
    / 2.0
)

volume_component = np.tanh(
    market["volume_shock_z"]
    .abs()
    .clip(upper=10)
    / 2.0
)

market["realized_impact_score_v2"] = (
    0.70 * return_component +
    0.30 * volume_component
)


# =============================================================================
# MERGE NEWS
# =============================================================================

market_label_cols = [
    "Date",
    "symbol",
    "Close",
    "Volume",
    "next_return",
    "next_nifty_return",
    "beta_60",
    "abnormal_next_return",
    "pre_vol_20d",
    "pre_vol_60d",
    "abnormal_return_z",
    "next_volume_ratio",
    "volume_shock_z",
    "realized_impact_score_v2",
]

df = news.merge(
    market[
        market_label_cols
    ],
    on=["Date", "symbol"],
    how="inner",
)


# =============================================================================
# CLEAN
# =============================================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan,
)

before = len(df)

df = df.dropna(
    subset=[
        "abnormal_next_return",
        "abnormal_return_z",
        "next_volume_ratio",
        "volume_shock_z",
        "realized_impact_score_v2",
    ]
).copy()

dropped = before - len(df)

df = (
    df.sort_values(["Date", "symbol"])
      .reset_index(drop=True)
)


# =============================================================================
# LABELS
# =============================================================================

q50 = df[
    "realized_impact_score_v2"
].quantile(0.50)

q80 = df[
    "realized_impact_score_v2"
].quantile(0.80)

df["impact_class_v2"] = np.select(
    [
        df["realized_impact_score_v2"] < q50,
        df["realized_impact_score_v2"] < q80,
    ],
    [
        "LOW",
        "MEDIUM",
    ],
    default="HIGH",
)

df["impact_class_v2_id"] = (
    df["impact_class_v2"]
    .map(
        {
            "LOW": 0,
            "MEDIUM": 1,
            "HIGH": 2,
        }
    )
    .astype(int)
)


# =============================================================================
# SAVE
# =============================================================================

df.to_parquet(
    OUT_FILE,
    index=False,
    compression="snappy",
)


# =============================================================================
# REPORT
# =============================================================================

print("\n" + "=" * 110)
print("V2 RESULT")
print("=" * 110)

print(
    f"Rows              : {len(df):,}"
)

print(
    f"Rows dropped      : {dropped:,}"
)

print(
    f"Stocks             : {df['symbol'].nunique()}"
)

print(
    f"Dates              : "
    f"{df['Date'].min()} -> {df['Date'].max()}"
)

market_stocks = set(
    market["symbol"].unique()
)

news_stocks = set(
    df["symbol"].unique()
)

print("\nMissing stocks from market universe:")
print(
    sorted(market_stocks - news_stocks)
)

print("\nImpact classes:")
print(
    df["impact_class_v2"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nImpact classes (%):")
print(
    (
        df["impact_class_v2"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    ).to_string()
)

print("\nContinuous impact score:")
print(
    df["realized_impact_score_v2"]
    .describe()
    .to_string()
)

print("\nAbnormal-return z-score:")
print(
    df["abnormal_return_z"]
    .describe()
    .to_string()
)

print("\nVolume-shock z-score:")
print(
    df["volume_shock_z"]
    .describe()
    .to_string()
)

print("\nMean absolute abnormal return by class:")
print(
    df.assign(
        abs_abnormal_return=
        df["abnormal_next_return"].abs()
    )
    .groupby("impact_class_v2")[
        "abs_abnormal_return"
    ]
    .mean()
    .sort_index()
    .to_string()
)

print("\nMean impact score by class:")
print(
    df.groupby("impact_class_v2")[
        "realized_impact_score_v2"
    ]
    .mean()
    .sort_index()
    .to_string()
)

print("\nMean volume shock by class:")
print(
    df.groupby("impact_class_v2")[
        "volume_shock_z"
    ]
    .mean()
    .sort_index()
    .to_string()
)

print(f"\nSaved:")
print(OUT_FILE)

print("\n" + "=" * 110)
print("STATUS: PASS — NEWS IMPACT V2 READY")
print("=" * 110)
