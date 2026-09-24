from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "/workspace/data/features/nse_features.parquet"
)

START_DATE = "2025-01-01"

TOP_KS = [1, 3, 5, 10]

HOLD_DAYS = 5

TRANSACTION_COST_BPS = 10.0
SLIPPAGE_BPS = 5.0


def metrics(returns):

    returns = pd.Series(
        returns
    ).dropna()

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

    cagr = (
        equity.iloc[-1]
        ** (1.0 / years)
        - 1.0
        if years > 0
        else np.nan
    )

    volatility = returns.std()

    sharpe = (
        np.sqrt(252.0 / HOLD_DAYS)
        * returns.mean()
        / volatility
        if volatility > 0
        else np.nan
    )

    downside = returns[
        returns < 0
    ]

    downside_std = downside.std()

    sortino = (
        np.sqrt(252.0 / HOLD_DAYS)
        * returns.mean()
        / downside_std
        if downside_std > 0
        else np.nan
    )

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
        - 1.0
    )

    return {
        "periods": len(returns),
        "total_return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "HitRate": (
            returns > 0
        ).mean(),
        "MaxDrawdown": drawdown.min(),
    }


def build_portfolio(
    df,
    score_column,
    top_k,
    phase,
):

    dates = sorted(
        df.timestamp.unique()
    )

    dates = dates[
        phase::HOLD_DAYS
    ]

    records = []

    previous_symbols = set()

    for date in dates:

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

        symbols = set(
            selected.symbol
        )

        turnover = (
            len(
                symbols
                - previous_symbols
            )
            / top_k
        )

        gross_return = (
            selected[
                "future_return_5d"
            ]
            .mean()
        )

        cost = (
            turnover
            * (
                TRANSACTION_COST_BPS
                + SLIPPAGE_BPS
            )
            / 10000.0
        )

        net_return = (
            gross_return
            - cost
        )

        records.append(
            {
                "date": date,
                "gross_return": gross_return,
                "cost": cost,
                "net_return": net_return,
                "turnover": turnover,
                "symbols": ",".join(
                    sorted(symbols)
                ),
            }
        )

        previous_symbols = symbols

    return pd.DataFrame(
        records
    )


def evaluate(
    name,
    df,
    score_column,
    top_k,
):

    phase_metrics = []

    all_returns = []

    for phase in range(
        HOLD_DAYS
    ):

        result = build_portfolio(
            df,
            score_column,
            top_k,
            phase,
        )

        if len(result) == 0:
            continue

        m = metrics(
            result.net_return
        )

        phase_metrics.append(
            m
        )

        all_returns.extend(
            result.net_return.tolist()
        )

    if not phase_metrics:
        return None

    # Average metrics across
    # the five independent phases.
    output = {}

    for key in [
        "total_return",
        "CAGR",
        "Sharpe",
        "Sortino",
        "HitRate",
        "MaxDrawdown",
    ]:

        values = [
            x[key]
            for x in phase_metrics
            if np.isfinite(x[key])
        ]

        output[key] = (
            np.mean(values)
            if values
            else np.nan
        )

    return output


def main():

    print("=" * 70)
    print(
        "REALISTIC PORTFOLIO BACKTEST"
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
    print("Rows:", len(df))
    print(
        "Days:",
        df.timestamp.nunique(),
    )
    print(
        "Symbols:",
        df.symbol.nunique(),
    )

    # ---------------------------------------------------------
    # RANDOM
    # ---------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    df["random_score"] = 0.0

    for timestamp, idx in df.groupby(
        "timestamp"
    ).groups.items():

        df.loc[
            idx,
            "random_score",
        ] = rng.random(
            len(idx)
        )

    # ---------------------------------------------------------
    # MOMENTUM
    # ---------------------------------------------------------

    df["momentum20"] = (
        df["return_20d"]
    )

    # ---------------------------------------------------------
    # 5D MOMENTUM
    # ---------------------------------------------------------

    df["momentum5"] = (
        df["return_5d"]
    )

    strategies = [
        (
            "RANDOM",
            "random_score",
        ),
        (
            "20D MOMENTUM",
            "momentum20",
        ),
        (
            "5D MOMENTUM",
            "momentum5",
        ),
    ]

    rows = []

    for name, score in strategies:

        print()
        print("=" * 70)
        print(name)
        print("=" * 70)

        for top_k in TOP_KS:

            result = evaluate(
                name,
                df,
                score,
                top_k,
            )

            print()
            print(
                f"TOP {top_k}"
            )

            for key, value in result.items():

                print(
                    f"{key:16s}: "
                    f"{value:.6f}"
                )

            rows.append(
                {
                    "Strategy": name,
                    "TopK": top_k,
                    **result,
                }
            )

    summary = pd.DataFrame(
        rows
    )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
