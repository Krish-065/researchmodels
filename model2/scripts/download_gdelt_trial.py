from pathlib import Path
from datetime import date, timedelta
import io
import re
import zipfile
import requests
import pandas as pd

ROOT = Path.home() / "24DIT065" / "model2"

ENTITIES = ROOT / "01_raw" / "news" / "news_entities.csv"

OUT = ROOT / "02_interim" / "news" / "gdelt_trial"

OUT.mkdir(parents=True, exist_ok=True)

START = date(2019, 1, 1)
END = date(2019, 1, 3)

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


def norm(x):
    return (
        str(x)
        .strip()
        .lower()
    )


def make_patterns():

    entities = pd.read_csv(ENTITIES)

    patterns = []

    for _, row in entities.iterrows():

        symbol = str(row["symbol"]).strip()

        aliases = [
            x.strip()
            for x in str(row["aliases"]).split("|")
            if x.strip()
        ]

        for alias in aliases:

            pattern = re.compile(
                r"(?<![A-Za-z])"
                + re.escape(alias.lower())
                + r"(?![A-Za-z])"
            )

            patterns.append(
                (
                    symbol,
                    alias,
                    pattern,
                )
            )

    return patterns


PATTERNS = make_patterns()


def process_day(day):

    datestr = day.strftime("%Y%m%d")

    url = (
        f"https://data.gdeltproject.org/"
        f"gkg/{datestr}.gkg.csv.zip"
    )

    print()
    print("=" * 90)
    print("PROCESSING:", datestr)
    print("=" * 90)
    print(url)

    r = requests.get(
        url,
        timeout=180,
    )

    print("HTTP:", r.status_code)
    print("Bytes:", len(r.content))

    if r.status_code != 200:

        raise RuntimeError(
            f"Download failed: "
            f"{datestr}, HTTP={r.status_code}"
        )

    with zipfile.ZipFile(
        io.BytesIO(r.content)
    ) as z:

        names = z.namelist()

        if not names:
            raise RuntimeError(
                f"Empty ZIP: {datestr}"
            )

        csv_name = names[0]

        raw = z.read(csv_name)

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

    print("GKG rows:", len(df))

    org_text = (
        df["ORGANIZATIONS"]
        .fillna("")
        .map(norm)
    )

    pieces = []

    for symbol, alias, pattern in PATTERNS:

        mask = org_text.str.contains(
            pattern,
            regex=True,
            na=False,
        )

        if mask.any():

            part = df.loc[mask].copy()

            part["symbol"] = symbol
            part["matched_alias"] = alias
            part["match_type"] = "organization"

            pieces.append(part)

    if pieces:

        result = pd.concat(
            pieces,
            ignore_index=True,
        )

        result = result.drop_duplicates(
            subset=[
                "DATE",
                "SOURCEURLS",
                "symbol",
            ]
        ).reset_index(drop=True)

    else:

        result = pd.DataFrame(
            columns=COLUMNS
            + [
                "symbol",
                "matched_alias",
                "match_type",
            ]
        )

    output = (
        OUT
        / f"{datestr}_matched.parquet"
    )

    result.to_parquet(
        output,
        index=False,
    )

    print("Relevant records:", len(result))
    print(
        "Unique URLs:",
        result["SOURCEURLS"].nunique()
        if len(result)
        else 0,
    )

    if len(result):

        print()
        print(
            result["symbol"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print()
    print("Saved:", output)


current = START

while current <= END:

    process_day(current)

    current += timedelta(days=1)


print()
print("=" * 90)
print("GDELT 3-DAY TRIAL COMPLETE")
print("=" * 90)

files = sorted(
    OUT.glob("*_matched.parquet")
)

print("Output files:", len(files))

print()
for p in files:
    print(p)

print("=" * 90)
