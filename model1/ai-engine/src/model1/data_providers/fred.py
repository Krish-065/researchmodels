from __future__ import annotations

import requests

from model1.settings import get_fred_api_key


FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


class FREDProvider:
    def __init__(self) -> None:
        self.api_key = get_fred_api_key()

    def get_observations(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:

        params = {
            "api_key": self.api_key,
            "series_id": series_id,
            "file_type": "json",
            "sort_order": "asc",
        }

        if start_date:
            params["observation_start"] = start_date

        if end_date:
            params["observation_end"] = end_date

        response = requests.get(
            FRED_URL,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()
