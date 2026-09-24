from __future__ import annotations

import os
from typing import Any

import requests


class AlphaVantageProvider:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")

        if not self.api_key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY is not configured.")

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        params = {
            **params,
            "apikey": self.api_key,
        }

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        if "Error Message" in data:
            raise RuntimeError(data["Error Message"])

        if "Note" in data:
            raise RuntimeError(
                f"Alpha Vantage API limit/message: {data['Note']}"
            )

        return data

    def daily(
        self,
        symbol: str,
        outputsize: str = "full",
    ) -> dict[str, Any]:
        return self._request(
            {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": outputsize,
            }
        )
