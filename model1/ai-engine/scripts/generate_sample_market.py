from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("/workspace/data/raw/sample_nifty50.parquet")


def main() -> None:
    rng = np.random.default_rng(42)

    timestamps = pd.date_range(
        "2020-01-01",
        periods=1000,
        freq="D",
        tz="UTC",
    )

    returns = rng.normal(
        loc=0.0003,
        scale=0.012,
        size=len(timestamps),
    )

    close = 10000 * np.exp(np.cumsum(returns))

    open_price = close * (
        1 + rng.normal(0, 0.003, len(close))
    )

    high = np.maximum(
        open_price,
        close,
    ) * (
        1 + rng.uniform(0, 0.008, len(close))
    )

    low = np.minimum(
        open_price,
        close,
    ) * (
        1 - rng.uniform(0, 0.008, len(close))
    )

    volume = rng.integers(
        1_000_000,
        10_000_000,
        len(close),
    )

    df = pd.DataFrame(
        {
            "symbol": "NIFTY50",
            "timestamp": timestamps,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT,
        index=False,
    )

    print(f"Generated {len(df):,} rows")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
