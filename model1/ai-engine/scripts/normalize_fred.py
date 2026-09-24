from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RAW_DIR = Path("/workspace/data/external/fred")
OUTPUT_DIR = Path("/workspace/data/processed/fred")


def normalize_file(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())

    observations = payload["data"]["observations"]

    rows = []

    for observation in observations:
        value = observation["value"]

        if value == ".":
            continue

        rows.append(
            {
                "date": observation["date"],
                "value": float(value),
            }
        )

    df = pd.DataFrame(rows)

    df["date"] = pd.to_datetime(df["date"], utc=True)

    series_id = path.stem

    df["series"] = series_id

    return df[
        [
            "series",
            "date",
            "value",
        ]
    ].sort_values("date")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.json"))

    if not files:
        raise RuntimeError("No FRED JSON files found.")

    all_data = []

    for path in files:
        print(f"Normalizing {path.name}")

        df = normalize_file(path)

        output = OUTPUT_DIR / f"{path.stem}.parquet"

        df.to_parquet(
            output,
            index=False,
        )

        all_data.append(df)

        print(
            f"  rows={len(df):,} "
            f"first={df['date'].min()} "
            f"last={df['date'].max()}"
        )

    combined = pd.concat(all_data, ignore_index=True)

    combined_output = OUTPUT_DIR / "fred_all.parquet"

    combined.to_parquet(
        combined_output,
        index=False,
    )

    print()
    print(f"Combined rows: {len(combined):,}")
    print(f"Output: {combined_output}")
    print()
    print("FRED NORMALIZATION: SUCCESS")


if __name__ == "__main__":
    main()
