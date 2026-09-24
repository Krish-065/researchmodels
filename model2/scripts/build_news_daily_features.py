from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path.home() / "24DIT065" / "model2"

INPUT = (
    ROOT
    / "02_interim"
    / "news"
    / "gdelt_news_clean.parquet"
)

OUT = (
    ROOT
    / "03_features"
    / "news"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 100)
print("MODEL 2 — DAILY STOCK NEWS FEATURES")
print("=" * 100)


news = pd.read_parquet(
    INPUT
)


# ============================================================
# Trading date
# ============================================================

news["Date"] = (
    pd.to_datetime(
        news["published_at"],
        utc=True,
    )
    .dt.tz_convert(
        "Asia/Kolkata"
    )
    .dt.normalize()
    .dt.tz_localize(None)
)


# ============================================================
# Daily aggregation
# ============================================================

g = news.groupby(
    [
        "Date",
        "symbol",
    ],
    sort=False,
)


daily = g.agg(
    news_count=(
        "article_id",
        "nunique",
    ),

    news_mean_sentiment=(
        "sentiment_score",
        "mean",
    ),

    news_std_sentiment=(
        "sentiment_score",
        "std",
    ),

    news_max_positive=(
        "sentiment_score",
        "max",
    ),

    news_min_negative=(
        "sentiment_score",
        "min",
)

).reset_index()


# ============================================================
# Sentiment counts
# ============================================================

counts = (
    news.assign(
        positive=(
            news["sentiment_score"]
            > 0.05
        ).astype(int),

        negative=(
            news["sentiment_score"]
            < -0.05
        ).astype(int),

        neutral=(
            (
                news["sentiment_score"]
                >= -0.05
            )
            &
            (
                news["sentiment_score"]
                <= 0.05
            )
        ).astype(int),
    )
    .groupby(
        [
            "Date",
            "symbol",
        ]
    )
    [
        [
            "positive",
            "negative",
            "neutral",
        ]
    ]
    .sum()
    .reset_index()
    .rename(
        columns={
            "positive":
                "news_positive_count",

            "negative":
                "news_negative_count",

            "neutral":
                "news_neutral_count",
        }
    )
)


daily = daily.merge(
    counts,
    on=[
        "Date",
        "symbol",
    ],
    how="left",
)


# ============================================================
# Rolling news pressure
# ============================================================

daily = daily.sort_values(
    [
        "symbol",
        "Date",
    ]
)


for period in [
    3,
    5,
    10,
]:

    daily[
        f"news_count_{period}d"
    ] = (
        daily
        .groupby("symbol")
        ["news_count"]
        .transform(
            lambda x:
            x.rolling(
                period,
                min_periods=1,
            ).sum()
        )
    )

    daily[
        f"news_sentiment_{period}d"
    ] = (
        daily
        .groupby("symbol")
        ["news_mean_sentiment"]
        .transform(
            lambda x:
            x.rolling(
                period,
                min_periods=1,
            ).mean()
        )
    )


# ============================================================
# News intensity
# ============================================================

daily["news_intensity"] = (
    np.log1p(
        daily["news_count"]
    )
)


daily["news_sentiment_pressure"] = (
    daily["news_mean_sentiment"]
    * daily["news_intensity"]
)


# ============================================================
# Fill only aggregation-produced values
# ============================================================

numeric_cols = [
    c
    for c in daily.columns
    if c not in [
        "Date",
        "symbol",
    ]
]


daily[numeric_cols] = (
    daily[numeric_cols]
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
    .fillna(0.0)
)


# ============================================================
# Save
# ============================================================

output = (
    OUT
    / "news_daily_stock_features.parquet"
)


daily.to_parquet(
    output,
    index=False,
)


# ============================================================
# Audit
# ============================================================

print()
print("=" * 100)
print("NEWS DAILY FEATURES")
print("=" * 100)

print(
    "Rows:",
    len(daily),
)

print(
    "Stocks:",
    daily["symbol"].nunique(),
)

print(
    "Dates:",
    daily["Date"].min(),
    "->",
    daily["Date"].max(),
)

print(
    "Columns:",
    len(daily.columns),
)

print()
print("Columns:")

for c in daily.columns:
    print(" ", c)


print()
print("News count statistics:")

print(
    daily["news_count"]
    .describe()
)


print()
print("Non-zero news rows:",
      int(
          (daily["news_count"] > 0)
          .sum()
      )
)


print()
print("Saved:")
print(output)


print()
print("=" * 100)

assert daily["symbol"].nunique() >= 48
assert daily["Date"].notna().all()
assert daily.duplicated(
    [
        "Date",
        "symbol",
    ]
).sum() == 0

print(
    "STATUS: PASS — DAILY NEWS FEATURES READY"
)

print("=" * 100)
