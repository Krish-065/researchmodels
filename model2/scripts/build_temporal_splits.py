from pathlib import Path
import pandas as pd

ROOT = Path.home() / "24DIT065" / "model2"

INPUT = ROOT / "04_model" / "datasets" / "model_dataset_daily.parquet"
OUT = ROOT / "04_model" / "splits"

OUT.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("MODEL 2 — BUILD TEMPORAL TRAIN / VALIDATION / TEST SPLITS")
print("=" * 100)

df = pd.read_parquet(INPUT)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "symbol"]).reset_index(drop=True)

# ------------------------------------------------------------------
# Temporal boundaries
# ------------------------------------------------------------------

TRAIN_END = pd.Timestamp("2023-12-31")
VALID_END = pd.Timestamp("2025-12-31")
TEST_END = pd.Timestamp("2026-08-25")

train = df[df["Date"] <= TRAIN_END].copy()

valid = df[
    (df["Date"] > TRAIN_END) &
    (df["Date"] <= VALID_END)
].copy()

test = df[
    (df["Date"] > VALID_END) &
    (df["Date"] <= TEST_END)
].copy()

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

train_path = OUT / "train.parquet"
valid_path = OUT / "validation.parquet"
test_path = OUT / "test.parquet"

train.to_parquet(train_path, index=False)
valid.to_parquet(valid_path, index=False)
test.to_parquet(test_path, index=False)

# ------------------------------------------------------------------
# Audit
# ------------------------------------------------------------------

print()
print("TRAIN")
print("Rows       :", len(train))
print("Dates      :", train["Date"].min(), "->", train["Date"].max())
print("Stocks     :", train["symbol"].nunique())

print()
print("VALIDATION")
print("Rows       :", len(valid))
print("Dates      :", valid["Date"].min(), "->", valid["Date"].max())
print("Stocks     :", valid["symbol"].nunique())

print()
print("TEST")
print("Rows       :", len(test))
print("Dates      :", test["Date"].min(), "->", test["Date"].max())
print("Stocks     :", test["symbol"].nunique())

print()
print("Rows total :", len(train) + len(valid) + len(test))
print("Original   :", len(df))

# ------------------------------------------------------------------
# Leakage / integrity checks
# ------------------------------------------------------------------

assert len(train) > 0
assert len(valid) > 0
assert len(test) > 0

assert train["Date"].max() < valid["Date"].min()
assert valid["Date"].max() < test["Date"].min()

assert train["symbol"].nunique() == 49
assert valid["symbol"].nunique() == 49
assert test["symbol"].nunique() == 49

assert train["Date"].max() <= TRAIN_END
assert valid["Date"].min() > TRAIN_END
assert valid["Date"].max() <= VALID_END
assert test["Date"].min() > VALID_END
assert test["Date"].max() <= TEST_END

assert (
    train.duplicated(["Date", "symbol"]).sum() == 0
)

assert (
    valid.duplicated(["Date", "symbol"]).sum() == 0
)

assert (
    test.duplicated(["Date", "symbol"]).sum() == 0
)

print()
print("=" * 100)
print("SPLITS SAVED")
print("=" * 100)

print(train_path)
print(valid_path)
print(test_path)

print()
print("STATUS: PASS — TEMPORAL SPLITS READY")
print("=" * 100)
