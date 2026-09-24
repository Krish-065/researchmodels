from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "/workspace/data/features/nse_features.parquet"
)

START_DATE = "2025-01-01"

HOLD_DAYS = 5

TOP_KS = [1, 3, 5, 10]

SEED = 42


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()

    drawdown = (
        equity / running_max
        - 1.0
    )

    return float(drawdown.min())


def performance(
    returns: pd.Series,
) -> dict:

    returns = returns.dropna()

    if len(returns) == 0:
        return {}

    equity = (
        1.0 + returns
    ).cumprod()

    total_return = (
        equity.iloc[-1] - 1.0
    )

    years = (
        len(returns)
        * HOLD_DAYS
        / 252.0
    )

    if years > 0:
        cagr = (
            equity.iloc[-1]
            ** (1.0 / years)
            - 1.0
        )
    else:
        cagr = np.nan

    mean_return = returns.mean()

    std_return = returns.std()

    sharpe = (
        np.sqrt(252.0 / HOLD_DAYS)
        * mean_return
        / std_return
        if std_return > 0
        else np.nan
    )

    hit_rate = (
        returns > 0
    ).mean()

    mdd = max_drawdown(
        equity
    )

    return {
        "periods": len(returns),
        "total_return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "hit_rate": hit_rate,
        "max_drawdown": mdd,
    }


def build_random_scores(
    df: pd.DataFrame,
) -> pd.Series:

    rng = np.random.default_rng(
        SEED
    )

    scores = np.empty(
        len(df),
        dtype=float,
    )

    for timestamp, idx in df.groupby(
        "timestamp"
    ).groups.items():

        scores[idx] = rng.random(
            len(idx)
        )

    return pd.Series(
        scores,
        index=df.index,
    )


def backtest(
    df: pd.DataFrame,
    score_column: str,
    top_k: int,
    phase: int,
):

    dates = sorted(
        df["timestamp"].unique()
    )

    # Select one non-overlapping stream.
    rebalance_dates = dates[
        phase::HOLD_DAYS
    ]

    results = []

    for date in rebalance_dates:

        day = df[
            df.timestamp == date
        ].copy()

        if len(day) < top_k:
            continue

        day = day.sort_values(
            score_column,
            ascending=False,
        )

        selected = day.head(
            top_k
        )

        portfolio_return = (
            selected[
                "future_return_5d"
            ]
            .mean()
        )

        results.append(
            {
                "date": date,
                "return": portfolio_return,
            }
        )

    if not results:
        return pd.Series(
            dtype=float
        )

    result = pd.DataFrame(
        results
    )

    return pd.Series(
        result["return"].values,
        index=pd.to_datetime(
            result["date"]
        ),
    )


def evaluate_strategy(
    name: str,
    df: pd.DataFrame,
    score_column: str,
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    phase_results = []

    for top_k in TOP_KS:

        all_phase_metrics = []

        for phase in range(
            HOLD_DAYS
        ):

            returns = backtest(
                df,
                score_column,
                top_k,
                phase,
            )

            metrics = performance(
                returns
            )

            all_phase_metrics.append(
                metrics
            )

        # Pool all 5 phases.
        #
        # They are separate non-overlapping
        # portfolio streams.
        #
        # We report the average metrics so
        # no single phase determines the result.

        summary = {}

        keys = [
            "total_return",
            "CAGR",
            "Sharpe",
            "hit_rate",
            "max_drawdown",
        ]

        for key in keys:

            values = [
                m[key]
                for m in all_phase_metrics
                if key in m
                and np.isfinite(m[key])
            ]

            summary[key] = (
                np.mean(values)
                if values
                else np.nan
            )

        print()
        print(
            f"TOP {top_k}"
        )

        print(
            "Average total return :",
            f"{summary['total_return']:.6f}",
        )

        print(
            "Average CAGR         :",
            f"{summary['CAGR']:.6f}",
        )

        print(
            "Average Sharpe       :",
            f"{summary['Sharpe']:.6f}",
        )

        print(
            "Average hit rate     :",
            f"{summary['hit_rate']:.6f}",
        )

        print(
            "Average max drawdown :",
            f"{summary['max_drawdown']:.6f}",
        )

        phase_results.append(
            {
                "top_k": top_k,
                **summary,
            }
        )

    return pd.DataFrame(
        phase_results
    )


def main():

    print("=" * 70)
    print(
        "NON-OVERLAPPING 5-DAY PORTFOLIO BACKTEST"
    )
    print("=" * 70)

    df = pd.read_parquet(
        INPUT
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df[
        df.timestamp >= START_DATE
    ].copy()

    df = df.sort_values(
        [
            "timestamp",
            "symbol",
        ]
    ).reset_index(drop=True)

    print()
    print("TEST DATA")
    print("-" * 70)

    print(
        "Rows   :",
        len(df),
    )

    print(
        "Days   :",
        df.timestamp.nunique(),
    )

    print(
        "Symbols:",
        df.symbol.nunique(),
    )

    print(
        "Period :",
        df.timestamp.min(),
        "->",
        df.timestamp.max(),
    )

    # --------------------------------------------------------
    # RANDOM
    # --------------------------------------------------------

    df["random_score"] = (
        build_random_scores(df)
    )

    random_result = evaluate_strategy(
        "RANDOM BASELINE",
        df,
        "random_score",
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    df["momentum_score"] = (
        df["return_20d"]
    )

    momentum_result = evaluate_strategy(
        "20-DAY MOMENTUM",
        df,
        "momentum_score",
    )

    # --------------------------------------------------------
    # CROSS-SECTIONAL RETURN
    # --------------------------------------------------------

    df["return_5d_score"] = (
        df["return_5d"]
    )

    return5_result = evaluate_strategy(
        "5-DAY MOMENTUM",
        df,
        "return_5d_score",
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, result in [
        (
            "RANDOM",
            random_result,
        ),
        (
            "20D MOMENTUM",
            momentum_result,
        ),
        (
            "5D MOMENTUM",
            return5_result,
        ),
    ]:

        print()
        print(name)

        print(
            result.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
