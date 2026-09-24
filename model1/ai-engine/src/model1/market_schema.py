from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketColumns:
    """Canonical column names used throughout Model 1."""

    symbol: str = "symbol"
    timestamp: str = "timestamp"

    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"

    volume: str = "volume"

    # Optional fields that some sources provide.
    adjusted_close: str = "adjusted_close"


REQUIRED_COLUMNS = (
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
