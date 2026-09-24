from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.home() / "24DIT065" / "model2"

RELATED = ROOT / "01_raw" / "yahoo" / "historical"
OUT = ROOT / "03_features" / "related"

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# RELATED ASSET DEFINITIONS
# ============================================================

ASSETS = {
    "NIFTY50": "INDEX_NSEI_daily.parquet",
    "INDIAVIX": "INDEX_INDIAVIX_daily.parquet",
    "SP500": "INDEX_GSPC_daily.parquet",
    "NASDAQ": "INDEX_IXIC_daily.parquet",
    "DOW": "INDEX_DJI_daily.parquet",
    "NIKKEI": "INDEX_N225_daily.parquet",
    "HANGSENG": "INDEX_HSI_daily.parquet",
    "GOLD": "GC_F_daily.parquet",
    "CRUDE": "CL_F_daily.parquet",
    "USDINR": "USDINR_X_daily.parquet",
    "BTC": "BTC_USD_daily.parquet",
    "ETH": "ETH_USD_daily.parquet",
}


# ============================================================
# HELPERS
# ============================================================

def find_column(df, names):

    # Exact match first
    for name in names:
        if name in df.columns:
            return name

    # Case-insensitive match
    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    # Handle flattened Yahoo columns such as:
    # ('Date', '')
    # ('Close', '^NSEI')
    for c in df.columns:

        text = str(c).lower()

        for name in names:

            if name.lower() in text:
                return c

    return None


def load_asset(name, filename):

    path = RELATED / filename

    print("Loading:", name)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing related asset: {path}"
        )

    df = pd.read_parquet(path)

    if len(df) == 0:
        raise ValueError(
            f"Empty related asset: {name}"
        )

    # --------------------------------------------------------
    # Find Date
    # --------------------------------------------------------

    date_col = find_column(
        df,
        ["Date", "Datetime", "Timestamp"]
    )

    if date_col is None:
        raise ValueError(
            f"Could not find Date column in {name}. "
            f"Columns={list(df.columns)}"
        )

    # --------------------------------------------------------
    # Find Close
    # --------------------------------------------------------

    close_col = find_column(
        df,
        ["Close"]
    )

    if close_col is None:
        raise ValueError(
            f"Could not find Close column in {name}. "
            f"Columns={list(df.columns)}"
        )

    # --------------------------------------------------------
    # Find Adj Close if available
    # --------------------------------------------------------

    adj_col = find_column(
        df,
        ["Adj Close", "Adjusted Close"]
    )

    # --------------------------------------------------------
    # Create clean dataframe
    # --------------------------------------------------------

    out = pd.DataFrame()

    out["Date"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    out["Close"] = pd.to_numeric(
        df[close_col],
        errors="coerce"
    )

    if adj_col is not None:

        out["Adj Close"] = pd.to_numeric(
            df[adj_col],
            errors="coerce"
        )

    else:

        out["Adj Close"] = out["Close"]

    # Remove invalid dates/prices
    out = out.dropna(
        subset=["Date", "Close"]
    )

    out = out[
        out["Close"] > 0
    ]

    # Remove duplicate dates
    out = (
        out.sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Use adjusted close where available
    # --------------------------------------------------------

    price = out["Adj Close"].copy()

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    out["return_1d"] = price.pct_change(1)
    out["return_3d"] = price.pct_change(3)
    out["return_5d"] = price.pct_change(5)
    out["return_10d"] = price.pct_change(10)
    out["return_20d"] = price.pct_change(20)

    # --------------------------------------------------------
    # Log return
    # --------------------------------------------------------

    out["log_return_1d"] = np.log(
        price / price.shift(1)
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    for period in [5, 10, 20, 50]:

        out[f"momentum_{period}"] = (
            price / price.shift(period) - 1
        )

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    for period in [5, 20, 50, 200]:

        ma = price.rolling(period).mean()

        out[f"sma_{period}"] = ma

        out[f"price_vs_sma_{period}"] = (
            price / ma - 1
        )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    for period in [5, 10, 20, 50]:

        out[f"volatility_{period}d"] = (
            out["return_1d"]
            .rolling(period)
            .std()
            * np.sqrt(252)
        )

    # --------------------------------------------------------
    # Rolling high / low
    # --------------------------------------------------------

    for period in [20, 50, 200]:

        rolling_high = (
            price.rolling(period).max()
        )

        rolling_low = (
            price.rolling(period).min()
        )

        out[f"distance_high_{period}"] = (
            price / rolling_high - 1
        )

        out[f"distance_low_{period}"] = (
            price / rolling_low - 1
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = price.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    out["rsi_14"] = (
        100 - (100 / (1 + rs))
    )

    # --------------------------------------------------------
    # Prefix all feature columns
    # --------------------------------------------------------

    feature_cols = [
        c for c in out.columns
        if c not in ["Date", "Close", "Adj Close"]
    ]

    rename = {
        c: f"{name.lower()}_{c}"
        for c in feature_cols
    }

    out = out.rename(
        columns=rename
    )

    # Keep a simple current price too
    out = out.rename(
        columns={
            "Close": f"{name.lower()}_close",
            "Adj Close": f"{name.lower()}_adj_close",
        }
    )

    return out


# ============================================================
# MAIN
# ============================================================

print("=" * 100)
print("MODEL 2 — RELATED ASSET FEATURE ENGINEERING")
print("=" * 100)

frames = []

for name, filename in ASSETS.items():

    df = load_asset(
        name,
        filename
    )

    print(
        f"  rows={len(df):5d} "
        f"{df['Date'].min().date()} -> "
        f"{df['Date'].max().date()}"
    )

    frames.append(df)


# ============================================================
# MERGE ALL RELATED ASSETS BY DATE
# ============================================================

print()
print("Merging related assets...")

related = frames[0]

for frame in frames[1:]:

    related = related.merge(
        frame,
        on="Date",
        how="outer"
    )


related = (
    related
    .sort_values("Date")
    .drop_duplicates("Date")
    .reset_index(drop=True)
)


# ============================================================
# Date range
# ============================================================

related = related[
    (related["Date"] >= "2019-01-01") &
    (related["Date"] <= "2026-09-01")
].copy()


# ============================================================
# Save
# ============================================================

output = (
    OUT /
    "related_asset_features_daily.parquet"
)

related.to_parquet(
    output,
    index=False
)


print()
print("=" * 100)
print("RELATED FEATURE DATASET CREATED")
print("=" * 100)

print("Rows       :", len(related))
print("Columns    :", len(related.columns))
print("Date start :", related["Date"].min())
print("Date end   :", related["Date"].max())

print()
print("Saved:")
print(output)

print()
print("=" * 100)
print("STATUS: PASS")
print("=" * 100)
