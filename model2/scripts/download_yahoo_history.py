import pandas as pd
import yfinance as yf
from pathlib import Path

ROOT = Path.home() / "24DIT065" / "model2"

UNIVERSE = ROOT / "01_raw" / "market" / "experiment1_universe.csv"
OUT = ROOT / "01_raw" / "yahoo" / "historical"

OUT.mkdir(parents=True, exist_ok=True)

START = "2019-01-01"
END = "2026-09-02"

u = pd.read_csv(UNIVERSE)

symbols = (
    u["symbol"]
    .astype(str)
    .str.strip()
    .tolist()
)

tickers = [s + ".NS" for s in symbols]

print("=" * 80)
print("MODEL 2 — YAHOO HISTORICAL DATA")
print("=" * 80)
print("Stocks :", len(tickers))
print("Start  :", START)
print("End    :", END)
print()

data = yf.download(
    tickers=tickers,
    start=START,
    end=END,
    interval="1d",
    auto_adjust=False,
    actions=True,
    group_by="ticker",
    threads=True,
    progress=True,
    repair=True,
)

print()

failed = []
saved = []

for ticker in tickers:

    symbol = ticker.replace(".NS", "")

    if ticker not in data.columns.get_level_values(0):
        failed.append(symbol)
        print("MISSING:", symbol)
        continue

    df = data[ticker].copy().reset_index()

    # Remove completely empty rows
    df = df.dropna(subset=["Close"], how="all")

    if len(df) == 0:
        failed.append(symbol)
        print("EMPTY  :", symbol)
        continue

    df["symbol"] = symbol

    path = OUT / f"{symbol}_daily.parquet"

    df.to_parquet(path, index=False)

    saved.append(symbol)

    print(f"OK {symbol:15s} {len(df):5d} rows")

print()
print("=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print("Requested :", len(symbols))
print("Saved     :", len(saved))
print("Failed    :", len(failed))

if failed:
    print()
    print("FAILED SYMBOLS:")
    for x in failed:
        print(" ", x)

print("=" * 80)

if failed:
    raise SystemExit(1)
