from dataclasses import dataclass


@dataclass(frozen=True)
class OHLCVContract:
    """Contract for market OHLCV data."""

    symbol_column: str = "symbol"
    timestamp_column: str = "timestamp"

    open_column: str = "open"
    high_column: str = "high"
    low_column: str = "low"
    close_column: str = "close"

    volume_column: str = "volume"
