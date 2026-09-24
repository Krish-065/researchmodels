from pathlib import Path


# /workspace inside the Docker container.
PROJECT_ROOT = Path("/workspace")

DATA_ROOT = PROJECT_ROOT / "data"

RAW_DATA = DATA_ROOT / "raw"
PROCESSED_DATA = DATA_ROOT / "processed"
FEATURE_DATA = DATA_ROOT / "features"
EXTERNAL_DATA = DATA_ROOT / "external"
CATALOG_DATA = DATA_ROOT / "catalog"

CONFIG_ROOT = PROJECT_ROOT / "config"

EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"


def ensure_directories() -> None:
    """Create required research directories."""

    directories = [
        RAW_DATA,
        PROCESSED_DATA,
        FEATURE_DATA,
        EXTERNAL_DATA,
        CATALOG_DATA,
        EXPERIMENT_ROOT,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
