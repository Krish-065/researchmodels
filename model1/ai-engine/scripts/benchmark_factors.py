from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("/workspace/data/features/nse_features.parquet")


def rank_cs(df, col):
    return df.groupby("timestamp")[col].rank(pct=True)


def evaluate(df, score, name, top_k=5):

    x = df[
        ["timestamp", "symbol", "future_return_5d"]
    ].copy()

    x["score"] = score.values

    x["rank"] = (
        x.groupby("timestamp")["score"]
        .rank(ascending=False, method="first")
    )

    top = x[x["rank"] <= top_k]

    daily = (
        top.groupby("timestamp")["future_return_5d"]
        .mean()
        .sort_index()
    )

    equity = (1 + daily).cumprod()

    total_return = equity.iloc[-1] - 1

    years = len(daily) / 252

    cagr = (
        equity.iloc[-1] ** (1 / years) - 1
        if years > 0
        else np.nan
    )

    sharpe = (
        np.sqrt(252 / 5)
        * daily.mean()
        / daily.std()
        if daily.std() > 0
        else np.nan
    )

    downside = daily[daily < 0].std()

    sortino = (
        np.sqrt(252 / 5)
        * daily.mean()
        / downside
        if downside > 0
        else np.nan
    )

    drawdown = equity / equity.cummax() - 1

    hit_rate = (daily > 0).mean()

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print("Top K          :", top_k)
    print("Total return   :", total_return)
    print("CAGR           :", cagr)
    print("Sharpe         :", sharpe)
    print("Sortino        :", sortino)
    print("Hit rate       :", hit_rate)
    print("Max drawdown   :", drawdown.min())


def main():

    print("=" * 70)
    print("MULTI-FACTOR CROSS-SECTIONAL BENCHMARK")
    print("=" * 70)

    df = pd.read_parquet(INPUT)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values(
        ["timestamp", "symbol"]
    ).reset_index(drop=True)

    test = df[
        df["timestamp"] >= "2025-01-01"
    ].copy()

    print()
    print("Test rows   :", len(test))
    print("Test symbols:", test.symbol.nunique())

    # ---------------------------------------------------------
    # Cross-sectional ranks
    # ---------------------------------------------------------

    factors = {}

    for col in [
        "return_5d",
        "return_20d",
        "return_60d",
        "return_1d",
        "close_sma20_ratio",
        "close_sma50_ratio",
        "close_sma200_ratio",
        "rsi_14",
        "volume_zscore",
    ]:

        factors[col] = rank_cs(test, col)

    # ---------------------------------------------------------
    # Individual factors
    # ---------------------------------------------------------

    for col, score in factors.items():

        evaluate(
            test,
            score,
            f"FACTOR: {col}",
            top_k=5,
        )

    # ---------------------------------------------------------
    # Multi-factor composite
    # ---------------------------------------------------------

    composite = (
        0.35 * factors["return_20d"]
        + 0.20 * factors["return_60d"]
        + 0.15 * factors["return_5d"]
        + 0.10 * factors["close_sma50_ratio"]
        + 0.10 * factors["close_sma200_ratio"]
        + 0.10 * factors["volume_zscore"]
    )

    evaluate(
        test,
        composite,
        "MULTI-FACTOR COMPOSITE",
        top_k=5,
    )


if __name__ == "__main__":
    main()
