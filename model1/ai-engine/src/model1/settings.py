from __future__ import annotations

import os


def get_required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is not configured."
        )

    return value


def get_fred_api_key() -> str:
    return get_required_env("FRED_API_KEY")


def get_alpha_vantage_api_key() -> str:
    return get_required_env("ALPHA_VANTAGE_API_KEY")
