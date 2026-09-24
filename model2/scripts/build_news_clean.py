from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path.home() / "24DIT065" / "model2"

NEWS_DIR = (
    ROOT
    / "02_interim"
    / "news"
    / "gdelt_daily"
)

OUT_DIR = (
    ROOT
    / "02_interim"
    / "news"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GDELT GKG columns used
# ============================================================

COLS = [
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
    "symbol",
    "matched_alias",
    "match_type",
]


# ============================================================
# Helpers
# ============================================================

def parse_tone(value):

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if not text:
        return np.nan

    try:
        first = text.split(",")[0]
        return float(first)
    except Exception:
        return np.nan


def parse_date(value):

    text = str(value).strip()

    # GKG 2.0 style:
    # YYYYMMDD
    #
    # GKG 2.1 style:
    # YYYYMMDDHHMMSS

    if len(text) >= 14:
        return pd.to_datetime(
            text[:14],
            format="%Y%m%d%H%M%S",
            errors="coerce",
            utc=True,
        )

    return pd.to_datetime(
        text[:8],
        format="%Y%m%d",
        errors="coerce",
        utc=True,
    )


# ============================================================
# Main
# ============================================================

print("=" * 100)
print("MODEL 2 — CLEAN GDELT NEWS")
print("=" * 100)

files = sorted(
    NEWS_DIR.glob(
        "*_matched.parquet"
    )
)

print("Input files:", len(files))

if not files:
    raise SystemExit(
        "No GDELT daily files found."
    )


frames = []

for i, path in enumerate(
    files,
    1,
):

    df = pd.read_parquet(
        path
    )

    if len(df) == 0:
        continue

    frames.append(
        df
    )

    if i % 250 == 0:
        print(
            f"Loaded {i}/{len(files)} files"
        )


if not frames:
    raise SystemExit(
        "All GDELT files were empty."
    )


news = pd.concat(
    frames,
    ignore_index=True,
)


print()
print("Raw matched records:", len(news))


# ============================================================
# Date / timestamp
# ============================================================

news["published_at"] = (
    news["DATE"]
    .map(parse_date)
)


# Remove invalid timestamps.
news = news.dropna(
    subset=[
        "published_at",
        "symbol",
    ]
).copy()


# ============================================================
# Normalize URL
# ============================================================

news["url"] = (
    news["SOURCEURLS"]
    .fillna("")
    .astype(str)
    .str.split("<UDIV>")
    .str[0]
    .str.strip()
)


# ============================================================
# Normalize symbol
# ============================================================

news["symbol"] = (
    news["symbol"]
    .astype(str)
    .str.strip()
)


# ============================================================
# GDELT Tone
# ============================================================

news["gdelt_tone"] = (
    news["TONE"]
    .map(parse_tone)
)


# ============================================================
# Basic sentiment transformation
#
# GDELT Tone is approximately centered around zero:
# negative = negative tone
# positive = positive tone.
#
# Keep original tone AND normalized value.
# ============================================================

news["sentiment_score"] = (
    news["gdelt_tone"]
    .fillna(0.0)
    / 10.0
)

news["sentiment_score"] = (
    news["sentiment_score"]
    .clip(
        -1.0,
        1.0,
    )
)


news["sentiment_positive"] = (
    news["sentiment_score"] > 0.05
).astype("int8")


news["sentiment_negative"] = (
    news["sentiment_score"] < -0.05
).astype("int8")


news["sentiment_neutral"] = (
    (
        news["sentiment_score"]
        >= -0.05
    )
    &
    (
        news["sentiment_score"]
        <= 0.05
    )
).astype("int8")


# ============================================================
# Remove obvious malformed records
# ============================================================

news = news[
    news["symbol"].ne("")
].copy()


# ============================================================
# Deduplicate
#
# Same article can occur:
# - through multiple URLs
# - through multiple GKG records
# - with several matching aliases.
#
# Keep one symbol/article/date record.
# ============================================================

news = (
    news
    .drop_duplicates(
        subset=[
            "published_at",
            "url",
            "symbol",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# Article uniqueness
# ============================================================

news["article_id"] = (
    news["url"]
    .replace(
        "",
        np.nan,
    )
)

missing_url = (
    news["article_id"]
    .isna()
)

# If URL isn't available, retain GKG identity.
news.loc[
    missing_url,
    "article_id",
] = (
    "gkg_"
    + news.index.astype(str)
)


# ============================================================
# Keep useful columns
# ============================================================

keep = [
    "published_at",
    "article_id",
    "url",
    "symbol",
    "matched_alias",
    "match_type",
    "gdelt_tone",
    "sentiment_score",
    "sentiment_positive",
    "sentiment_negative",
    "sentiment_neutral",
    "THEMES",
    "LOCATIONS",
    "PERSONS",
    "ORGANIZATIONS",
    "SOURCES",
]


news = news[
    [
        c
        for c in keep
        if c in news.columns
    ]
]


news = (
    news
    .sort_values(
        [
            "published_at",
            "symbol",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# Save
# ============================================================

output = (
    OUT_DIR
    / "gdelt_news_clean.parquet"
)

news.to_parquet(
    output,
    index=False,
)


# ============================================================
# Audit
# ============================================================

print()
print("=" * 100)
print("CLEAN NEWS DATASET")
print("=" * 100)

print(
    "Records       :",
    len(news),
)

print(
    "Unique articles:",
    news["article_id"].nunique(),
)

print(
    "Stocks        :",
    news["symbol"].nunique(),
)

print(
    "Date start    :",
    news["published_at"].min(),
)

print(
    "Date end      :",
    news["published_at"].max(),
)


all_symbols = set(
    pd.read_csv(
        ROOT
        / "01_raw"
        / "market"
        / "experiment1_universe.csv"
    )["symbol"]
    .astype(str)
    .str.strip()
)

news_symbols = set(
    news["symbol"]
)

missing_symbols = sorted(
    all_symbols - news_symbols
)

print()
print("Stocks with news:",
      len(news_symbols))

print(
    "Stocks without news:",
    len(missing_symbols),
)

print(
    missing_symbols
    if missing_symbols
    else "NONE"
)


print()
print("Records by stock:")

print(
    news["symbol"]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("Sentiment statistics:")

print(
    news[
        "sentiment_score"
    ].describe()
)


print()
print("Missing values:")

print(
    news.isna()
    .sum()
    .sort_values(
        ascending=False
    )
    .head(20)
)


print()
print("Saved:")
print(output)

print()
print("=" * 100)

assert len(news) > 100000
assert news["symbol"].nunique() >= 48
assert news["published_at"].notna().all()
assert news["sentiment_score"].notna().all()

print(
    "STATUS: PASS — CLEAN NEWS DATASET READY"
)

print("=" * 100)
