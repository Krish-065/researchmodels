from pathlib import Path
import pandas as pd

ROOT = Path.home() / "24DIT065" / "model2"

UNIVERSE = ROOT / "01_raw" / "market" / "experiment1_universe.csv"
HIST = ROOT / "01_raw" / "yahoo" / "historical"
OUT = ROOT / "02_interim" / "yahoo" / "master"

OUT.mkdir(parents=True, exist_ok=True)

symbols = (
    pd.read_csv(UNIVERSE)["symbol"]
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)

print("=" * 100)
print("MODEL 2 — BUILD MASTER STOCK DATASET")
print("=" * 100)

frames = []

for symbol in symbols:

    path = HIST / f"{symbol}_daily.parquet"

    print("Loading:", symbol)

    df = pd.read_parquet(path)

    # Keep only useful columns
    keep = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    ]

    keep = [c for c in keep if c in df.columns]

    df = df[keep].copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["symbol"] = symbol

    frames.append(df)

master = pd.concat(
    frames,
    ignore_index=True
)

# Sort
master = master.sort_values(
    ["Date", "symbol"]
).reset_index(drop=True)

# Remove duplicate stock/date combinations
master = master.drop_duplicates(
    subset=["Date", "symbol"],
    keep="last"
)

# Numeric conversion
numeric = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]

for col in numeric:

    if col in master.columns:

        master[col] = pd.to_numeric(
            master[col],
            errors="coerce"
        )

# Save
parquet_path = OUT / "master_stocks_daily.parquet"
csv_path = OUT / "master_stocks_daily.csv"

master.to_parquet(
    parquet_path,
    index=False
)

master.to_csv(
    csv_path,
    index=False
)

print()
print("=" * 100)
print("MASTER DATASET CREATED")
print("=" * 100)

print("Rows       :", len(master))
print("Stocks     :", master["symbol"].nunique())
print("Date start :", master["Date"].min())
print("Date end   :", master["Date"].max())

print()
print("Rows per stock:")
print(
    master.groupby("symbol")
    .size()
    .describe()
)

print()
print("Saved:")
print(parquet_path)
print(csv_path)

print()
print("STATUS: PASS")
print("=" * 100)
