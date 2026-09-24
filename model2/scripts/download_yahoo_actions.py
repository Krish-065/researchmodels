import pandas as pd
import yfinance as yf
from pathlib import Path
import time

ROOT = Path.home() / "24DIT065" / "model2"

UNIVERSE = ROOT / "01_raw" / "market" / "experiment1_universe.csv"
OUT = ROOT / "01_raw" / "yahoo" / "actions"

OUT.mkdir(parents=True, exist_ok=True)

universe = pd.read_csv(UNIVERSE)

symbols = (
    universe["symbol"]
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)

print("=" * 80)
print("MODEL 2 — YAHOO CORPORATE ACTIONS")
print("=" * 80)
print("Universe:", len(symbols))
print("Output  :", OUT)
print("=" * 80)

failed = []
saved = []

for i, symbol in enumerate(symbols, 1):

    ticker = symbol + ".NS"

    print(f"[{i:02d}/{len(symbols)}] {ticker}")

    try:
        t = yf.Ticker(ticker)

        actions = t.actions

        if actions is None or len(actions) == 0:
            print("    No corporate-action rows")
            continue

        actions = actions.reset_index()

        # Normalize column names
        actions.columns = [str(c).strip() for c in actions.columns]

        actions["symbol"] = symbol

        parquet_path = OUT / f"{symbol}_actions.parquet"
        csv_path = OUT / f"{symbol}_actions.csv"

        actions.to_parquet(parquet_path, index=False)
        actions.to_csv(csv_path, index=False)

        saved.append(symbol)

        dividends = 0
        splits = 0

        if "Dividends" in actions.columns:
            dividends = int(
                (pd.to_numeric(
                    actions["Dividends"],
                    errors="coerce"
                ).fillna(0) != 0).sum()
            )

        if "Stock Splits" in actions.columns:
            splits = int(
                (pd.to_numeric(
                    actions["Stock Splits"],
                    errors="coerce"
                ).fillna(0) != 0).sum()
            )

        print(
            f"    rows={len(actions)} "
            f"dividends={dividends} "
            f"splits={splits}"
        )

    except Exception as e:

        print("    ERROR:", repr(e))
        failed.append(symbol)

    # Avoid hammering Yahoo
    time.sleep(0.5)

print()
print("=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print("Requested:", len(symbols))
print("Saved    :", len(saved))
print("Failed   :", len(failed))

if failed:
    print()
    print("FAILED SYMBOLS:")
    for symbol in failed:
        print(" ", symbol)

print("=" * 80)

if failed:
    raise SystemExit(1)

print("STATUS: PASS")
