from pathlib import Path

ROOT = Path.home() / "24DIT065" / "model2"

START_DATE = "2019-01-01"
END_DATE = "2026-08-25"

RAW_NEWS = ROOT / "01_raw" / "news"
INTERIM_NEWS = ROOT / "02_interim" / "news"
FEATURE_NEWS = ROOT / "03_features" / "news"

UNIVERSE = (
    ROOT
    / "01_raw"
    / "market"
    / "experiment1_universe.csv"
)

for p in [
    RAW_NEWS,
    INTERIM_NEWS,
    FEATURE_NEWS,
]:
    p.mkdir(parents=True, exist_ok=True)
