from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from model1.data_providers.fred import FREDProvider


OUTPUT_DIR = Path("/workspace/data/external/fred")


SERIES = {
    "FEDFUNDS": "Federal Funds Rate",
    "CPIAUCSL": "Consumer Price Index",
    "UNRATE": "Unemployment Rate",
    "DGS10": "10-Year Treasury Rate",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    provider = FREDProvider()

    for series_id, description in SERIES.items():
        print(f"Downloading {series_id} - {description}")

        data = provider.get_observations(
            series_id,
            start_date="2000-01-01",
        )

        output = OUTPUT_DIR / f"{series_id}.json"

        payload = {
            "series_id": series_id,
            "description": description,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }

        output.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        observations = len(data.get("observations", []))

        print(f"  observations: {observations}")
        print(f"  saved: {output}")


if __name__ == "__main__":
    main()
