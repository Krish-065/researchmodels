from pathlib import Path
import pandas as pd

ROOT = Path.home() / "24DIT065" / "model2"

UNIVERSE = ROOT / "01_raw" / "market" / "experiment1_universe.csv"
HIST = ROOT / "01_raw" / "yahoo" / "historical"

symbols = (
    pd.read_csv(UNIVERSE)["symbol"]
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)

print("=" * 100)
print("MODEL 2 — 49 STOCK HISTORICAL DATA AUDIT")
print("=" * 100)

print("Universe:", len(symbols))
print()

failed = []

for symbol in symbols:

    path = HIST / f"{symbol}_daily.parquet"

    if not path.exists():
        print(f"FAIL {symbol:15s} FILE MISSING")
        failed.append(symbol)
        continue

    try:
        df = pd.read_parquet(path)

        required = {
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        }

        missing = required - set(df.columns)

        if missing:
            print(
                f"FAIL {symbol:15s} "
                f"missing={sorted(missing)}"
            )
            failed.append(symbol)
            continue

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        if df["Date"].isna().all():
            print(f"FAIL {symbol:15s} INVALID DATES")
            failed.append(symbol)
            continue

        # Check OHLC
        bad_ohlc = (
            (df["High"] < df["Low"]) |
            (df["High"] < df["Open"]) |
            (df["High"] < df["Close"]) |
            (df["Low"] > df["Open"]) |
            (df["Low"] > df["Close"])
        ).sum()

        # Check duplicate dates
        duplicates = df["Date"].duplicated().sum()

        # Check missing close
        missing_close = df["Close"].isna().sum()

        # Check non-positive prices
        bad_price = (
            pd.to_numeric(
                df["Close"],
                errors="coerce"
            ) <= 0
        ).sum()

        problems = []

        if bad_ohlc:
            problems.append(f"bad_ohlc={bad_ohlc}")

        if duplicates:
            problems.append(f"duplicate_dates={duplicates}")

        if missing_close:
            problems.append(f"missing_close={missing_close}")

        if bad_price:
            problems.append(f"bad_price={bad_price}")

        if problems:

            print(
                f"FAIL {symbol:15s} "
                + " ".join(problems)
            )

            failed.append(symbol)
            continue

        print(
            f"PASS {symbol:15s} "
            f"rows={len(df):5d} "
            f"{df['Date'].min().date()} -> "
            f"{df['Date'].max().date()}"
        )

    except Exception as e:

        print(
            f"FAIL {symbol:15s} "
            f"ERROR={repr(e)}"
        )

        failed.append(symbol)

print()
print("=" * 100)

print("Failed:", len(failed))

if failed:

    print()
    print("FAILED SYMBOLS:")

    for symbol in failed:
        print(" ", symbol)

    raise SystemExit(1)

print()
print("STATUS: PASS — ALL 49 STOCK DATASETS VALID")
print("=" * 100)
