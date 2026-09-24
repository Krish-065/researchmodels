import pandas as pd
import numpy as np

p = "/workspace/data/features/nse_features.parquet"
df = pd.read_parquet(p)

print("=" * 70)
print("LEAKAGE / TEMPORAL FEATURE AUDIT")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nFEATURE CORRELATION WITH TARGET")

features = [
    "open","high","low","close","volume","adjusted_close",
    "return_1d","return_5d","return_20d","return_60d",
    "sma_5","sma_20","sma_50","sma_200",
    "close_sma5_ratio","close_sma20_ratio",
    "close_sma50_ratio","close_sma200_ratio",
    "ema_12","ema_26","macd","macd_signal",
    "rsi_14","volatility_20d","volatility_60d",
    "high_low_range","open_close_return",
    "volume_change","volume_zscore"
]

target = "target_up_5d"

corr = df[features + [target]].corr(numeric_only=True)[target].drop(target)
print(corr.sort_values(ascending=False).to_string())

print("\n" + "=" * 70)
print("TARGET / FUTURE-COLUMN CHECK")
print("=" * 70)

future_cols = [
    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "target_up_5d"
]

print(df[future_cols].isna().sum())

print("\nFuture columns:")
print(future_cols)

print("\n" + "=" * 70)
print("TIMESTAMP / FEATURE AVAILABILITY CHECK")
print("=" * 70)

for c in features:
    if c in df:
        print(f"{c:25s} dtype={str(df[c].dtype):12s} nulls={df[c].isna().sum()}")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
