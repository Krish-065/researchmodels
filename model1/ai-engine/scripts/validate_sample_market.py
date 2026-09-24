from pathlib import Path

import pandas as pd

from model1.market_validation import validate_ohlcv


INPUT = Path("/workspace/data/raw/sample_nifty50.parquet")


def main() -> None:
    df = pd.read_parquet(INPUT)

    print("Rows before validation:", len(df))

    validated = validate_ohlcv(df)

    print("Rows after validation :", len(validated))
    print("Columns               :", list(validated.columns))
    print(
        "Time range            :",
        validated["timestamp"].min(),
        "→",
        validated["timestamp"].max(),
    )

    print()
    print("MARKET DATA VALIDATION: SUCCESS")


if __name__ == "__main__":
    main()
