from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.home() / "24DIT065" / "model2"

MASTER = (
    ROOT
    / "02_interim"
    / "yahoo"
    / "master"
    / "master_stocks_daily.parquet"
)

RELATED = ROOT / "01_raw" / "yahoo" / "historical"

OUT = ROOT / "03_features" / "market"

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df, period=14):

    prev_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()


def add_stock_features(df):

    df = df.sort_values("Date").copy()

    close = df["Close"]
    volume = df["Volume"]

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    df["return_1d"] = close.pct_change(1)
    df["return_3d"] = close.pct_change(3)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)

    # --------------------------------------------------------
    # Log return
    # --------------------------------------------------------

    df["log_return_1d"] = np.log(
        close / close.shift(1)
    )

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    for period in [5, 10, 20, 50, 100, 200]:

        df[f"sma_{period}"] = (
            close.rolling(period).mean()
        )

        df[f"ema_{period}"] = (
            close.ewm(
                span=period,
                adjust=False
            ).mean()
        )

    # --------------------------------------------------------
    # Distance from moving averages
    # --------------------------------------------------------

    for period in [20, 50, 200]:

        df[f"close_vs_sma_{period}"] = (
            close / df[f"sma_{period}"] - 1
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi_14"] = rsi(close, 14)

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = (
        df["macd"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std

    df["bb_width"] = (
        (df["bb_upper"] - df["bb_lower"])
        / bb_mid
    )

    df["bb_position"] = (
        (close - df["bb_lower"])
        / (df["bb_upper"] - df["bb_lower"])
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["atr_14"] = atr(df, 14)

    df["atr_pct"] = (
        df["atr_14"] / close
    )

    # --------------------------------------------------------
    # Daily range
    # --------------------------------------------------------

    df["high_low_range"] = (
        (df["High"] - df["Low"])
        / close
    )

    df["open_close_range"] = (
        (df["Close"] - df["Open"])
        / df["Open"]
    )

    # --------------------------------------------------------
    # Gap
    # --------------------------------------------------------

    df["gap"] = (
        df["Open"]
        / close.shift(1)
        - 1
    )

    # --------------------------------------------------------
    # Rolling volatility
    # --------------------------------------------------------

    for period in [5, 10, 20, 50]:

        df[f"volatility_{period}d"] = (
            df["return_1d"]
            .rolling(period)
            .std()
            * np.sqrt(252)
        )

    # --------------------------------------------------------
    # Volume features
    # --------------------------------------------------------

    df["volume_change_1d"] = (
        volume.pct_change()
    )

    for period in [5, 20, 50]:

        df[f"volume_sma_{period}"] = (
            volume.rolling(period).mean()
        )

    df["volume_ratio_20"] = (
        volume / df["volume_sma_20"]
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    for period in [5, 10, 20, 50]:

        df[f"momentum_{period}"] = (
            close / close.shift(period) - 1
        )

    # --------------------------------------------------------
    # Rolling high / low
    # --------------------------------------------------------

    for period in [20, 50, 200]:

        rolling_high = (
            close.rolling(period).max()
        )

        rolling_low = (
            close.rolling(period).min()
        )

        df[f"distance_high_{period}"] = (
            close / rolling_high - 1
        )

        df[f"distance_low_{period}"] = (
            close / rolling_low - 1
        )

    return df


# ============================================================
# Load master stock dataset
# ============================================================

print("=" * 100)
print("MODEL 2 — STOCK FEATURE ENGINEERING")
print("=" * 100)

print("Loading:", MASTER)

master = pd.read_parquet(MASTER)

master["Date"] = pd.to_datetime(
    master["Date"]
)

master = master.sort_values(
    ["symbol", "Date"]
).reset_index(drop=True)

print("Input rows:", len(master))
print("Stocks:", master["symbol"].nunique())

# ============================================================
# Generate stock features
# ============================================================

frames = []

for symbol, group in master.groupby(
    "symbol",
    sort=True
):

    print("Processing:", symbol)

    features = add_stock_features(group)

    frames.append(features)

stock_features = pd.concat(
    frames,
    ignore_index=True
)

stock_features = stock_features.sort_values(
    ["Date", "symbol"]
).reset_index(drop=True)

# ============================================================
# Save
# ============================================================

output = OUT / "stock_features_daily.parquet"

stock_features.to_parquet(
    output,
    index=False
)

print()
print("=" * 100)
print("FEATURE DATASET CREATED")
print("=" * 100)

print("Rows       :", len(stock_features))
print("Stocks     :", stock_features["symbol"].nunique())
print("Date start :", stock_features["Date"].min())
print("Date end   :", stock_features["Date"].max())
print("Columns    :", len(stock_features.columns))

print()
print("Saved:")
print(output)

print()
print("STATUS: PASS")
print("=" * 100)
