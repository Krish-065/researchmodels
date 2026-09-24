from pathlib import Path
import pandas as pd
import re

ROOT = Path.home() / "24DIT065" / "model2"

GKG = (
    ROOT
    / "01_raw"
    / "news"
    / "gdelt_gkg"
    / "test"
    / "20260831.gkg.csv"
)

ENTITIES = (
    ROOT
    / "01_raw"
    / "news"
    / "news_entities.csv"
)

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


def normalize_text(x):
    return (
        str(x)
        .replace("\x00", " ")
        .strip()
        .lower()
    )


print("=" * 100)
print("GDELT SAMPLE — SAFE COMPANY MATCH AUDIT")
print("=" * 100)

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

        # Word-boundary match.
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


df = pd.read_csv(
    GKG,
    sep="\t",
    header=None,
    names=COLUMNS,
    skiprows=1,
    dtype=str,
    encoding="utf-8",
    on_bad_lines="skip",
    low_memory=False,
)

print("Total GKG records:", len(df))

org_text = (
    df["ORGANIZATIONS"]
    .fillna("")
    .map(normalize_text)
)

results = []

for symbol, alias, pattern in patterns:

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

        results.append(part)


if not results:
    raise SystemExit(
        "No company organization matches found."
    )

result = pd.concat(
    results,
    ignore_index=True,
)

# One source article may appear multiple times.
result = result.drop_duplicates(
    subset=[
        "DATE",
        "SOURCEURLS",
        "symbol",
    ]
).reset_index(drop=True)

print()
print("Relevant records:", len(result))
print("Unique URLs     :", result["SOURCEURLS"].nunique())
print(
    "Stocks matched  :",
    result["symbol"].nunique(),
)

print()
print("Matches by stock:")
print(
    result["symbol"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()
print("Top stocks by news count:")
print(
    result["symbol"]
    .value_counts()
    .head(15)
    .to_string()
)

out = (
    ROOT
    / "02_interim"
    / "news"
    / "gdelt_sample_matched_safe.parquet"
)

out.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_parquet(
    out,
    index=False,
)

print()
print("Saved:", out)
print()
print("=" * 100)
print("STATUS: PASS")
print("=" * 100)
