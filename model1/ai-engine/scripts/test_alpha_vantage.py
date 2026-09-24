from __future__ import annotations

import os

from model1.data_providers.alpha_vantage import AlphaVantageProvider


def main() -> None:
    provider = AlphaVantageProvider()

    print("=== ALPHA VANTAGE API TEST ===")

    data = provider.daily(
        symbol="IBM",
        outputsize="compact",
    )

    metadata = data.get("Meta Data", {})
    time_series = data.get("Time Series (Daily)", {})

    print("Symbol:", metadata.get("2. Symbol"))
    print("Rows:", len(time_series))

    if time_series:
        dates = sorted(time_series)

        print("First date:", dates[0])
        print("Last date :", dates[-1])

    print()
    print("ALPHA VANTAGE API: SUCCESS")


if __name__ == "__main__":
    main()
