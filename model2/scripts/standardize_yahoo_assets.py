from pathlib import Path
import pandas as pd
import re

ROOT = Path.home() / "24DIT065" / "model2"

HIST = ROOT / "01_raw" / "yahoo" / "historical"
OUT = ROOT / "02_interim" / "yahoo"

OUT.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("MODEL 2 — STANDARDIZE YAHOO HISTORICAL ASSETS")
print("=" * 90)

files = sorted(HIST.glob("*_daily.parquet"))

# Only process related assets here.
# The 49 individual stock files will be handled separately.
related_prefixes = (
    "INDEX_",
    "GC_",
    "CL_",
    "USDINR_",
    "BTC_",
    "ETH_",
)

files = [
    p for p in files
    if p.name.startswith(related_prefixes)
]

print("Files found:", len(files))
print()

failed = []
saved = []

for path in files:

    print("Processing:", path.name)

    try:
        df = pd.read_parquet(path)

        if df.empty:
            raise ValueError("Empty dataframe")

        # ------------------------------------------------------------
        # Normalize Yahoo's flattened column names
        # ------------------------------------------------------------

        new_columns = []

        for col in df.columns:

            col = str(col).strip()

            # Examples:
            # ('Date', '')
            # ('Close', '^NSEI')
            # ('Volume', 'BTC-USD')

            if col.startswith("(") and "," in col:

                parts = col.strip("()").split(",")

                first = parts[0].strip().strip("'").strip('"')
                second = parts[1].strip().strip("'").strip('"')

                if first == "Date":
                    new_columns.append("Date")
                else:
                    new_columns.append(first)

            else:
                new_columns.append(col)

        df.columns = new_columns

        # ------------------------------------------------------------
        # Ensure Date exists
        # ------------------------------------------------------------

        if "Date" not in df.columns:
            raise ValueError(
                f"No Date column after normalization. "
                f"Columns={df.columns.tolist()}"
            )

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.dropna(subset=["Date"])

        # ------------------------------------------------------------
        # Remove duplicate dates
        # ------------------------------------------------------------

        df = (
            df
            .sort_values("Date")
            .drop_duplicates(subset=["Date"], keep="last")
            .reset_index(drop=True)
        )

        # ------------------------------------------------------------
        # Standardize numeric columns
        # ------------------------------------------------------------

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "Dividends",
            "Stock Splits",
        ]

        for col in numeric_columns:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        # ------------------------------------------------------------
        # Create clean filename
        # ------------------------------------------------------------

        out_path = OUT / path.name

        df.to_parquet(
            out_path,
            index=False
        )

        saved.append(path.name)

        print(
            f"  PASS | rows={len(df):5d} | "
            f"{df['Date'].min().date()} -> "
            f"{df['Date'].max().date()}"
        )

    except Exception as e:

        print("  FAIL:", repr(e))
        failed.append(path.name)

    print()

print("=" * 90)
print("SUMMARY")
print("=" * 90)

print("Processed:", len(files))
print("Saved    :", len(saved))
print("Failed   :", len(failed))

if failed:

    print()
    print("FAILED FILES:")

    for x in failed:
        print(" ", x)

    raise SystemExit(1)

print()
print("STATUS: PASS")
print("Output:", OUT)
print("=" * 90)
