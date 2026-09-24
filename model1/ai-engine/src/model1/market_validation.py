from __future__ import annotations

import pandas as pd

from model1.market_schema import REQUIRED_COLUMNS


class MarketDataValidationError(ValueError):
    """Raised when market data violates the canonical contract."""


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate canonical OHLCV data.

    Returns a cleaned copy if validation succeeds.
    """

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise MarketDataValidationError(
            f"Missing required columns: {missing}"
        )

    result = df.copy()

    # Timestamp validation.
    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="raise",
    )

    # Numeric validation.
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

    # Basic price sanity checks.
    if (result["high"] < result["low"]).any():
        raise MarketDataValidationError(
            "Found rows where high < low."
        )

    if (result["open"] <= 0).any():
        raise MarketDataValidationError(
            "Found non-positive open prices."
        )

    if (result["high"] <= 0).any():
        raise MarketDataValidationError(
            "Found non-positive high prices."
        )

    if (result["low"] <= 0).any():
        raise MarketDataValidationError(
            "Found non-positive low prices."
        )

    if (result["close"] <= 0).any():
        raise MarketDataValidationError(
            "Found non-positive close prices."
        )

    if (result["volume"] < 0).any():
        raise MarketDataValidationError(
            "Found negative volume."
        )

    # Remove exact duplicate rows.
    result = result.drop_duplicates(
        subset=["symbol", "timestamp"]
    )

    # Sort chronologically.
    result = result.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)

    return result
