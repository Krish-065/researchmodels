from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from model1.data_providers.alpha_vantage import AlphaVantageProvider
from model1.market_validation import validate_ohlcv


RAW_ROOT = Path("/workspace/data/external/alpha_vantage")
PROCESSED_ROOT = Path("/workspace/data/processed/market")

SYMBOLS = [
    "RELIANCE.BSE",
]


def main() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    provider = AlphaVantageProvider()

    for symbol in SYMBOLS:
        print()
        print("=" * 60)
        print("Downloading:", symbol)
        print("=" * 60)

        data = provider.daily(
            symbol=symbol,
            outputsize="full",
        )

        series = data.get("Time Series (Daily)", {})

        if not series:
            raise RuntimeError(
                f"No daily data returned for {symbol}. "
                f"Available keys: {list(data.keys())}"
            )

        raw_path = RAW_ROOT / f"{symbol.replace('.', '_')}.json"
        raw_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

        rows = []

        for timestamp, values in series.items():
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "open": values["1. open"],
                    "high": values["2. high"],
                    "low": values["3. low"],
                    "close": values["4. close"],
                    "volume": values["5. volume"],
                }
            )

        df = pd.DataFrame(rows)

        validated = validate_ohlcv(df)

        output_path = (
            PROCESSED_ROOT
            / f"{symbol.replace('.', '_')}_daily.parquet"
        )

        validated.to_parquet(
            output_path,
            index=False,
        )

        print("Rows      :", len(validated))
        print("First     :", validated["timestamp"].min())
        print("Last      :", validated["timestamp"].max())
        print("Output    :", output_path)
        print("Raw       :", raw_path)

    print()
    print("INDIAN MARKET DOWNLOAD: SUCCESS")


if __name__ == "__main__":
    main()
