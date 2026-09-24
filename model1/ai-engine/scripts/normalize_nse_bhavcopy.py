from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from model1.market_validation import validate_ohlcv


RAW_ROOT = Path("/workspace/data/raw/market/nse/bhavcopy")
OUTPUT_ROOT = Path("/workspace/data/processed/market")

OUTPUT_FILE = OUTPUT_ROOT / "nse_cm_ohlcv.parquet"


COLUMN_MAP = {
    "TckrSymb": "symbol",
    "TradDt": "timestamp",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "TtlTradgVol": "volume",
}


def read_zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        csv_files = [
            name
            for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise RuntimeError(f"No CSV found inside {path}")

        with z.open(csv_files[0]) as f:
            return pd.read_csv(
                f,
                dtype=str,
            )


def normalize_file(path: Path) -> pd.DataFrame:
    print(f"Reading: {path.name}")

    df = read_zip_csv(path)

    missing = [
        column
        for column in COLUMN_MAP
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{path.name}: missing columns {missing}"
        )

    # Keep normal equity instruments only.
    #
    # STK = equity/security-type records.
    # This prevents accidentally treating bonds/other
    # instruments as ordinary equities.
    if "FinInstrmTp" in df.columns:
        df = df[df["FinInstrmTp"].eq("STK")].copy()

    result = df[
        list(COLUMN_MAP.keys())
    ].rename(columns=COLUMN_MAP)

    # Convert types.
    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="coerce",
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    # Remove unusable rows.
    result = result.dropna(
        subset=[
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    # Validate against Model 1's canonical contract.
    result = validate_ohlcv(result)

    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        RAW_ROOT.glob("*.zip")
    )

    if not files:
        raise RuntimeError(
            f"No NSE bhavcopy ZIP files found in {RAW_ROOT}"
        )

    print("=" * 70)
    print("NSE CM NORMALIZATION")
    print("=" * 70)
    print("Files:", len(files))

    frames: list[pd.DataFrame] = []

    for path in files:
        frame = normalize_file(path)

        print(
            f"  rows={len(frame):,} "
            f"symbols={frame['symbol'].nunique():,}"
        )

        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = (
        combined
        .drop_duplicates(
            subset=["symbol", "timestamp"]
        )
        .sort_values(
            ["symbol", "timestamp"]
        )
        .reset_index(drop=True)
    )

    combined.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)
    print("Rows   :", f"{len(combined):,}")
    print("Symbols:", f"{combined['symbol'].nunique():,}")
    print(
        "First  :",
        combined["timestamp"].min(),
    )
    print(
        "Last   :",
        combined["timestamp"].max(),
    )
    print("Output :", OUTPUT_FILE)


if __name__ == "__main__":
    main()
