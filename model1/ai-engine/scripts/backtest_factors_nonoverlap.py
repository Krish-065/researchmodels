from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("/workspace/data/features/nse_features.parquet")


def cross_sectional_rank(df, column):
    return (
        df.groupby("timestamp")[column]
        .rank(pct=True)
    )


def make_scores(df):

    scores = pd.DataFrame(index=df.index)

    scores["return_5d"] = cross_sectional_rank(
        df, "return_5d"
    )

    scores["return_20d"] = cross_sectional_rank(
        df, "return_20d"
    )

    scores["return_60d"] = cross_sectional_rank(
        df, "return_60d"
    )

    scores["return_1d"] = cross_sectional_rank(
        df, "return_1d"
    )

    scores["sma20"] = cross_sectional_rank(
        df, "close_sma20_ratio"
    )

    scores["sma50"] = cross_sectional_rank(
        df, "close_sma50_ratio"
    )

    scores["sma200"] = cross_sectional_rank(
        df, "close_sma200_ratio"
    )

    scores["rsi"] = cross_sectional_rank(
        df, "rsi_14"
    )

    scores["volume"] = cross_sectional_rank(
        df, "volume_zscore"
    )

    return scores


def portfolio_backtest(
    df,
    score,
    top_k=5,
    holding_days=5,
):

    x = df[
        [
            "timestamp",
            "symbol",
            "future_return_5d",
        ]
    ].copy()

    x["score"] = score

    # ----------------------------------------------------------
    # One rebalance every 5 trading days.
    # ----------------------------------------------------------

    dates = (
        x["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    rebalance_dates = dates.iloc[
        ::holding_days
    ]

    x = x[
        x["timestamp"].isin(rebalance_dates)
    ].copy()

    # ----------------------------------------------------------
    # Select top K stocks.
    # ----------------------------------------------------------

    x["rank"] = (
        x.groupby("timestamp")["score"]
        .rank(
            ascending=False,
            method="first",
        )
    )

    selected = x[x["rank"] <= top_k].copy()

    # ----------------------------------------------------------
    # Portfolio return per rebalance.
    # ----------------------------------------------------------

    daily = (
        selected
        .groupby("timestamp")["future_return_5d"]
        .mean()
        .sort_index()
    )

    if len(daily) == 0:
        return None

    equity = (1 + daily).cumprod()

    total_return = equity.iloc[-1] - 1

    years = len(daily) * holding_days / 252

    cagr = (
        equity.iloc[-1] ** (1 / years) - 1
    )

    sharpe = (
        np.sqrt(252 / holding_days)
        * daily.mean()
        / daily.std()
        if daily.std() > 0
        else np.nan
    )

    downside = daily[daily < 0].std()

    sortino = (
        np.sqrt(252 / holding_days)
        * daily.mean()
        / downside
        if downside > 0
        else np.nan
    )

    drawdown = (
        equity / equity.cummax()
    ) - 1

    hit_rate = (
        daily > 0
    ).mean()

    return {
        "total_return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "HitRate": hit_rate,
        "MaxDrawdown": drawdown.min(),
        "periods": len(daily),
    }


def main():

    print("=" * 70)
    print("NON-OVERLAPPING FACTOR BENCHMARK")
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
    print("Rows   :", len(test))
    print("Symbols:", test["symbol"].nunique())

    scores = make_scores(test)

    strategies = {
        "5D MOMENTUM":
            scores["return_5d"],

        "20D MOMENTUM":
            scores["return_20d"],

        "60D MOMENTUM":
            scores["return_60d"],

        "1D MOMENTUM":
            scores["return_1d"],

        "SMA20":
            scores["sma20"],

        "SMA50":
            scores["sma50"],

        "SMA200":
            scores["sma200"],

        "RSI":
            scores["rsi"],

        "VOLUME":
            scores["volume"],
    }

    # ----------------------------------------------------------
    # Composite candidates
    # ----------------------------------------------------------

    strategies["COMPOSITE"] = (
        0.45 * scores["return_20d"]
        + 0.20 * scores["return_60d"]
        + 0.15 * scores["sma20"]
        + 0.10 * scores["rsi"]
        + 0.10 * scores["volume"]
    )

    strategies["MOMENTUM_ONLY"] = (
        0.65 * scores["return_20d"]
        + 0.35 * scores["return_60d"]
    )

    strategies["MOMENTUM_TREND"] = (
        0.50 * scores["return_20d"]
        + 0.25 * scores["return_60d"]
        + 0.25 * scores["sma20"]
    )

    results = []

    for name, score in strategies.items():

        result = portfolio_backtest(
            test,
            score,
            top_k=5,
            holding_days=5,
        )

        print()
        print("-" * 70)
        print(name)
        print("-" * 70)

        if result is None:
            print("No result")
            continue

        for key in [
            "total_return",
            "CAGR",
            "Sharpe",
            "Sortino",
            "HitRate",
            "MaxDrawdown",
        ]:
            print(
                f"{key:15s}: "
                f"{result[key]:.6f}"
            )

        results.append({
            "Strategy": name,
            **result,
        })

    print()
    print("=" * 70)
    print("FINAL RANKING")
    print("=" * 70)

    result_df = pd.DataFrame(results)

    print(
        result_df
        .sort_values(
            "Sharpe",
            ascending=False,
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
