from pathlib import Path
import pandas as pd

ROOT = Path("/workspace/data/raw/market/yfinance")
OUT = Path("/workspace/data/processed/market/nse_equity_daily.parquet")

files = sorted(ROOT.glob("*_daily.parquet"))

frames = []

for path in files:
    df = pd.read_parquet(path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Ensure numeric fields are numeric.
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    frames.append(df)

df = pd.concat(frames, ignore_index=True)

# Remove duplicate symbol/date observations.
df = (
    df.drop_duplicates(["symbol", "timestamp"])
      .sort_values(["symbol", "timestamp"])
      .reset_index(drop=True)
)

# Basic validity checks.
df = df[
    (df["open"] > 0)
    & (df["high"] > 0)
    & (df["low"] > 0)
    & (df["close"] > 0)
    & (df["volume"] >= 0)
]

if (df["high"] < df["low"]).any():
    raise RuntimeError("Found high < low rows.")

if (df["high"] < df["open"]).any():
    raise RuntimeError("Found high < open rows.")

if (df["high"] < df["close"]).any():
    raise RuntimeError("Found high < close rows.")

if (df["low"] > df["open"]).any():
    raise RuntimeError("Found low > open rows.")

if (df["low"] > df["close"]).any():
    raise RuntimeError("Found low > close rows.")

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)

print("=" * 70)
print("CANONICAL MARKET DATASET")
print("=" * 70)
print("Rows    :", f"{len(df):,}")
print("Symbols :", df["symbol"].nunique())
print("First   :", df["timestamp"].min())
print("Last    :", df["timestamp"].max())
print("Output  :", OUT)
