from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


INPUT = Path("/workspace/data/features/nse_features.parquet")


BASE_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "close_sma5_ratio",
    "close_sma20_ratio",
    "close_sma50_ratio",
    "close_sma200_ratio",
    "macd",
    "macd_signal",
    "rsi_14",
    "volatility_20d",
    "volatility_60d",
    "high_low_range",
    "open_close_return",
    "volume_change",
    "volume_zscore",
]


RANK_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "close_sma5_ratio",
    "close_sma20_ratio",
    "close_sma50_ratio",
    "close_sma200_ratio",
    "rsi_14",
    "volatility_20d",
    "volatility_60d",
    "volume_zscore",
]


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["timestamp", "symbol"]).copy()

    daily = df.groupby("timestamp")

    # ------------------------------------------------------------
    # Cross-sectional ranks
    # ------------------------------------------------------------

    for col in RANK_FEATURES:
        df[f"{col}_rank"] = (
            daily[col]
            .rank(pct=True)
        )

    # ------------------------------------------------------------
    # Cross-sectional z-scores
    # ------------------------------------------------------------

    for col in RANK_FEATURES:
        mean = daily[col].transform("mean")
        std = daily[col].transform("std")

        df[f"{col}_z"] = (
            (df[col] - mean)
            / std.replace(0, np.nan)
        )

    # ------------------------------------------------------------
    # Market state
    # ------------------------------------------------------------

    df["market_return_1d"] = daily["return_1d"].transform("mean")
    df["market_return_5d"] = daily["return_5d"].transform("mean")
    df["market_return_20d"] = daily["return_20d"].transform("mean")

    df["market_volatility"] = (
        daily["volatility_20d"].transform("mean")
    )

    df["breadth_1d"] = daily["return_1d"].transform(
        lambda x: (x > 0).mean()
    )

    df["breadth_5d"] = daily["return_5d"].transform(
        lambda x: (x > 0).mean()
    )

    df["breadth_sma20"] = daily["close_sma20_ratio"].transform(
        lambda x: (x > 0).mean()
    )

    df["breadth_sma50"] = daily["close_sma50_ratio"].transform(
        lambda x: (x > 0).mean()
    )

    df["breadth_sma200"] = daily["close_sma200_ratio"].transform(
        lambda x: (x > 0).mean()
    )

    # ------------------------------------------------------------
    # Relative strength
    # ------------------------------------------------------------

    df["relative_return_1d"] = (
        df["return_1d"] - df["market_return_1d"]
    )

    df["relative_return_5d"] = (
        df["return_5d"] - df["market_return_5d"]
    )

    df["relative_return_20d"] = (
        df["return_20d"] - df["market_return_20d"]
    )

    # ------------------------------------------------------------
    # Correct market regime
    #
    # Use historical rolling market volatility rather than
    # comparing a value against its own same-day median.
    # ------------------------------------------------------------

    market_vol = (
        df[["timestamp", "market_volatility"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )

    market_vol["market_vol_median_60d"] = (
        market_vol["market_volatility"]
        .rolling(60, min_periods=20)
        .median()
    )

    df = df.merge(
        market_vol[
            [
                "timestamp",
                "market_vol_median_60d",
            ]
        ],
        on="timestamp",
        how="left",
    )

    df["market_high_volatility"] = (
        df["market_volatility"]
        > df["market_vol_median_60d"]
    ).astype("int8")

    df["market_positive_regime"] = (
        df["market_return_20d"] > 0
    ).astype("int8")

    return df


def make_relevance_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert future 5-day return into cross-sectional relevance.

    Higher future return = higher relevance.

    Five relevance buckets:
        0 = bottom 20%
        1 = 20-40%
        2 = 40-60%
        3 = 60-80%
        4 = top 20%
    """

    df = df.copy()

    df["relevance"] = (
        df.groupby("timestamp")["future_return_5d"]
        .rank(
            pct=True,
            method="first",
        )
        .mul(5)
        .astype(int)
        .clip(upper=4)
    )

    return df


def clean_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


def make_groups(df: pd.DataFrame) -> list[int]:
    return (
        df.groupby("timestamp", sort=False)
        .size()
        .tolist()
    )


def daily_rank_metrics(
    df: pd.DataFrame,
    score: np.ndarray,
    name: str,
):
    x = df[
        [
            "timestamp",
            "symbol",
            "future_return_5d",
        ]
    ].copy()

    x["score"] = score

    x["rank"] = (
        x.groupby("timestamp")["score"]
        .rank(
            ascending=False,
            method="first",
        )
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    # ------------------------------------------------------------
    # Rank IC
    # ------------------------------------------------------------

    ic = (
        x.groupby("timestamp")
        .apply(
            lambda g: g["score"].corr(
                g["future_return_5d"],
                method="spearman",
            ),
            include_groups=False,
        )
        .dropna()
    )

    print()
    print("RANK IC")
    print("-" * 70)
    print("Mean IC        :", ic.mean())
    print("Median IC      :", ic.median())
    print("IC std         :", ic.std())
    print("Positive IC %  :", (ic > 0).mean())

    # ------------------------------------------------------------
    # Portfolio metrics
    # ------------------------------------------------------------

    for k in [1, 3, 5, 10]:

        top = x[x["rank"] <= k]

        daily_return = (
            top.groupby("timestamp")["future_return_5d"]
            .mean()
        )

        mean_return = daily_return.mean()
        std_return = daily_return.std()

        sharpe = (
            np.sqrt(252 / 5)
            * mean_return
            / std_return
            if std_return > 0
            else np.nan
        )

        hit_rate = (
            top["future_return_5d"] > 0
        ).mean()

        cumulative = (
            1 + daily_return
        ).cumprod()

        max_drawdown = (
            cumulative / cumulative.cummax() - 1
        ).min()

        print()
        print(f"TOP {k}")
        print("Mean 5d return :", mean_return)
        print("Std  5d return :", std_return)
        print("Approx Sharpe  :", sharpe)
        print("Hit rate       :", hit_rate)
        print("Max drawdown   :", max_drawdown)


def main():

    print("=" * 70)
    print("TRUE CROSS-SECTIONAL LEARNING-TO-RANK")
    print("=" * 70)

    df = pd.read_parquet(INPUT)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values(
        ["timestamp", "symbol"]
    ).reset_index(drop=True)

    print()
    print("BUILDING FEATURES...")

    df = add_cross_sectional_features(df)
    df = make_relevance_labels(df)

    # ------------------------------------------------------------
    # Feature list
    # ------------------------------------------------------------

    extra_rank = [
        f"{x}_rank"
        for x in RANK_FEATURES
    ]

    extra_z = [
        f"{x}_z"
        for x in RANK_FEATURES
    ]

    market_features = [
        "market_return_1d",
        "market_return_5d",
        "market_return_20d",
        "market_volatility",
        "breadth_1d",
        "breadth_5d",
        "breadth_sma20",
        "breadth_sma50",
        "breadth_sma200",
        "relative_return_1d",
        "relative_return_5d",
        "relative_return_20d",
        "market_positive_regime",
        "market_high_volatility",
    ]

    FEATURES = (
        BASE_FEATURES
        + extra_rank
        + extra_z
        + market_features
    )

    print("Base features :", len(BASE_FEATURES))
    print("Rank features :", len(extra_rank))
    print("Z features    :", len(extra_z))
    print("Market        :", len(market_features))
    print("Total         :", len(FEATURES))

    # ------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------

    train = df[
        df.timestamp < "2024-01-01"
    ].copy()

    valid = df[
        (df.timestamp >= "2024-01-01")
        & (df.timestamp < "2025-01-01")
    ].copy()

    test = df[
        df.timestamp >= "2025-01-01"
    ].copy()

    print()
    print("DATA")
    print("-" * 70)
    print("Train :", len(train))
    print("Valid :", len(valid))
    print("Test  :", len(test))

    print()
    print("DATES")
    print("-" * 70)
    print("Train:", train.timestamp.min(), "->", train.timestamp.max())
    print("Valid:", valid.timestamp.min(), "->", valid.timestamp.max())
    print("Test :", test.timestamp.min(), "->", test.timestamp.max())

    # ------------------------------------------------------------
    # Prepare matrices
    # ------------------------------------------------------------

    X_train = clean_features(train, FEATURES)
    X_valid = clean_features(valid, FEATURES)
    X_test = clean_features(test, FEATURES)

    y_train = train["relevance"].astype(int)
    y_valid = valid["relevance"].astype(int)

    train_groups = make_groups(train)
    valid_groups = make_groups(valid)

    print()
    print("RANKING GROUPS")
    print("-" * 70)
    print("Train groups:", len(train_groups))
    print("Valid groups:", len(valid_groups))
    print(
        "Train group sizes:",
        min(train_groups),
        "to",
        max(train_groups),
    )
    print(
        "Valid group sizes:",
        min(valid_groups),
        "to",
        max(valid_groups),
    )

    # ------------------------------------------------------------
    # LightGBM LambdaRank
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING LIGHTGBM LAMBDARANK")
    print("=" * 70)

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_at=[1, 3, 5, 10],
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        group=train_groups,
        eval_set=[
            (X_valid, y_valid),
        ],
        eval_group=[
            valid_groups,
        ],
        callbacks=[
            lgb.early_stopping(
                75,
                verbose=True,
            ),
        ],
    )

    # ------------------------------------------------------------
    # Test
    # ------------------------------------------------------------

    test_score = model.predict(
        X_test,
        num_iteration=model.best_iteration_,
    )

    daily_rank_metrics(
        test,
        test_score,
        "TEST — LIGHTGBM LAMBDARANK",
    )

    # ------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------

    output = Path(
        "/workspace/data/models/"
        "lightgbm_lambdarank.txt"
    )

    model.booster_.save_model(
        str(output)
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)
    print(output)


if __name__ == "__main__":
    main()
