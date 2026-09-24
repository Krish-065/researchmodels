from pathlib import Path
from datetime import date, timedelta
import io
import re
import zipfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests


# ============================================================
# PATHS
# ============================================================

ROOT = Path.home() / "24DIT065" / "model2"

ENTITIES = (
    ROOT
    / "01_raw"
    / "news"
    / "news_entities.csv"
)

OUT = (
    ROOT
    / "02_interim"
    / "news"
    / "gdelt_daily"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATE RANGE
# ============================================================

START = date(
    2019,
    1,
    1,
)

END = date(
    2026,
    8,
    25,
)


# ============================================================
# DOWNLOAD SETTINGS
# ============================================================

WORKERS = 4

TIMEOUT = 180

RETRIES = 4


# ============================================================
# GDELT GKG COLUMNS USED
# ============================================================

COLUMNS = [
    "DATE",
    "NUMARTS",
    "COUNTS",
    "THEMES",
    "LOCATIONS",
    "PERSONS",
    "ORGANIZATIONS",
    "TONE",
    "CAMEOEVENTIDS",
    "SOURCES",
    "SOURCEURLS",
]


# ============================================================
# LOAD ENTITY MAP
# ============================================================

entities = pd.read_csv(
    ENTITIES
)

patterns = []

for _, row in entities.iterrows():

    symbol = str(
        row["symbol"]
    ).strip()

    aliases = [
        x.strip()
        for x in str(
            row["aliases"]
        ).split("|")
        if x.strip()
    ]

    for alias in aliases:

        pattern = re.compile(
            r"(?<![A-Za-z])"
            + re.escape(
                alias.lower()
            )
            + r"(?![A-Za-z])"
        )

        patterns.append(
            (
                symbol,
                alias,
                pattern,
            )
        )


# ============================================================
# ONE DAY
# ============================================================

def process_day(day):

    datestr = day.strftime(
        "%Y%m%d"
    )

    output = (
        OUT
        / f"{datestr}_matched.parquet"
    )

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if (
        output.exists()
        and output.stat().st_size > 0
    ):
        return (
            datestr,
            "SKIP",
            0,
            0,
            0,
        )

    url = (
        "https://data.gdeltproject.org/"
        f"gkg/{datestr}.gkg.csv.zip"
    )

    last_error = None

    for attempt in range(
        1,
        RETRIES + 1,
    ):

        try:

            response = requests.get(
                url,
                timeout=TIMEOUT,
            )

            if response.status_code == 404:

                return (
                    datestr,
                    "NOFILE",
                    0,
                    0,
                    0,
                )

            response.raise_for_status()

            archive_bytes = (
                response.content
            )

            with zipfile.ZipFile(
                io.BytesIO(
                    archive_bytes
                )
            ) as z:

                members = [
                    x
                    for x in z.namelist()
                    if x.lower().endswith(".csv")
                ]

                if not members:
                    raise RuntimeError(
                        "No CSV inside ZIP"
                    )

                raw = z.read(
                    members[0]
                )

            # ------------------------------------------------
            # Read only the columns we need.
            #
            # Current GDELT daily GKG sample has 11 columns.
            # ------------------------------------------------

            df = pd.read_csv(
                io.BytesIO(raw),
                sep="\t",
                header=None,
                names=COLUMNS,
                skiprows=1,
                dtype=str,
                encoding="utf-8",
                on_bad_lines="skip",
                low_memory=False,
            )

            raw_rows = len(df)

            # ------------------------------------------------
            # Organization/entity matching
            # ------------------------------------------------

            organization_text = (
                df[
                    "ORGANIZATIONS"
                ]
                .fillna("")
                .astype(str)
                .str.lower()
            )

            pieces = []

            for (
                symbol,
                alias,
                pattern,
            ) in patterns:

                mask = (
                    organization_text
                    .str.contains(
                        pattern,
                        regex=True,
                        na=False,
                    )
                )

                if mask.any():

                    part = df.loc[
                        mask,
                        COLUMNS,
                    ].copy()

                    part[
                        "symbol"
                    ] = symbol

                    part[
                        "matched_alias"
                    ] = alias

                    part[
                        "match_type"
                    ] = "organization"

                    pieces.append(
                        part
                    )

            # ------------------------------------------------
            # Empty-day output
            # ------------------------------------------------

            if pieces:

                result = pd.concat(
                    pieces,
                    ignore_index=True,
                )

                # The same source may be associated with
                # multiple organization mentions.
                result = (
                    result
                    .drop_duplicates(
                        subset=[
                            "DATE",
                            "SOURCEURLS",
                            "symbol",
                        ]
                    )
                    .reset_index(
                        drop=True
                    )
                )

            else:

                result = pd.DataFrame(
                    columns=(
                        COLUMNS
                        + [
                            "symbol",
                            "matched_alias",
                            "match_type",
                        ]
                    )
                )

            # ------------------------------------------------
            # Save only relevant records
            # ------------------------------------------------

            result.to_parquet(
                output,
                index=False,
            )

            return (
                datestr,
                "OK",
                raw_rows,
                len(result),
                (
                    result["symbol"]
                    .nunique()
                    if len(result)
                    else 0
                ),
            )

        except Exception as exc:

            last_error = exc

            if (
                attempt
                < RETRIES
            ):

                time.sleep(
                    min(
                        30,
                        2 ** attempt,
                    )
                )

    return (
        datestr,
        "ERROR:"
        + type(
            last_error
        ).__name__,
        0,
        0,
        0,
    )


# ============================================================
# BUILD DATE LIST
# ============================================================

dates = []

current = START

while current <= END:

    dates.append(
        current
    )

    current += timedelta(
        days=1
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 100)
print(
    "MODEL 2 — GDELT HISTORICAL NEWS COLLECTION"
)
print("=" * 100)

print(
    "Start date :",
    START,
)

print(
    "End date   :",
    END,
)

print(
    "Days       :",
    len(dates),
)

print(
    "Workers    :",
    WORKERS,
)

print(
    "Output     :",
    OUT,
)

print("=" * 100)


results = []

with ThreadPoolExecutor(
    max_workers=WORKERS
) as executor:

    future_map = {
        executor.submit(
            process_day,
            d,
        ): d
        for d in dates
    }

    for i, future in enumerate(
        as_completed(
            future_map
        ),
        1,
    ):

        result = (
            future.result()
        )

        results.append(
            result
        )

        (
            datestr,
            status,
            raw_rows,
            matched_rows,
            stock_count,
        ) = result

        print(
            f"[{i:4d}/{len(dates)}] "
            f"{datestr} "
            f"{status:18s} "
            f"raw={raw_rows:7d} "
            f"matched={matched_rows:6d} "
            f"stocks={stock_count:2d}",
            flush=True,
        )


# ============================================================
# SUMMARY
# ============================================================

ok = sum(
    r[1] == "OK"
    for r in results
)

skipped = sum(
    r[1] == "SKIP"
    for r in results
)

nofile = sum(
    r[1] == "NOFILE"
    for r in results
)

errors = [
    r
    for r in results
    if str(
        r[1]
    ).startswith(
        "ERROR"
    )
]

raw_total = sum(
    r[2]
    for r in results
)

matched_total = sum(
    r[3]
    for r in results
)


summary = pd.DataFrame(
    results,
    columns=[
        "date",
        "status",
        "raw_rows",
        "matched_rows",
        "stocks",
    ],
)

summary = summary.sort_values(
    "date"
)

summary_path = (
    OUT
    / "download_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


print()
print("=" * 100)
print(
    "GDELT HISTORICAL DOWNLOAD SUMMARY"
)
print("=" * 100)

print(
    "Requested days :",
    len(dates),
)

print(
    "Processed OK   :",
    ok,
)

print(
    "Already existed:",
    skipped,
)

print(
    "No file        :",
    nofile,
)

print(
    "Errors         :",
    len(errors),
)

print(
    "Raw GKG rows   :",
    raw_total,
)

print(
    "Matched rows   :",
    matched_total,
)

print(
    "Output files   :",
    len(
        list(
            OUT.glob(
                "*_matched.parquet"
            )
        )
    ),
)

print()
print(
    "Summary:",
    summary_path,
)

if errors:

    print()
    print("FIRST ERRORS:")

    for error in sorted(
        errors
    )[:20]:

        print(
            " ",
            error
        )

print("=" * 100)

if errors:

    raise SystemExit(1)

print(
    "STATUS: PASS — HISTORICAL GDELT COLLECTION COMPLETE"
)
print("=" * 100)
