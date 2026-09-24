from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    path: str
    rows: int
    columns: list[str]
    first_timestamp: str
    last_timestamp: str
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_manifest(
    name: str,
    path: Path,
) -> DatasetManifest:

    df = pd.read_parquet(path)

    return DatasetManifest(
        name=name,
        path=str(path),
        rows=len(df),
        columns=list(df.columns),
        first_timestamp=str(df["timestamp"].min()),
        last_timestamp=str(df["timestamp"].max()),
        sha256=sha256_file(path),
    )
