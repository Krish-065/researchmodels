from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("/workspace/data/features/nse_features.parquet")


def rank_cs(df, col):
    return df.groupby("timestamp")[col].rank(pct=True)


def build_scores(df):
    s = pd.DataFrame(index=df.index)

    s["r20"] = rank_cs(df, "return_20d")
    s["r60"] = rank_cs(df, "return_60d")
    s["sma20"] = rank_cs(df, "close_sma20_ratio")
    s["rsi"] = rank_cs(df, "rsi_14")
    s["volume"] = rank_cs(df, "volume_zscore")

    # Candidate composite discovered in Step 1.
    s["composite"] = (
        0.45 * s["r20"]
        + 0.20 * s["r60"]
        + 0.15 * s["sma20"]
        + 0.10 * s["rsi"]
        + 0.10 * s["volume"]
    )

    return s


def evaluate_period(df, score, top_k=5, holding_days=5):

    dates = (
        df["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    rebalance_dates = dates.iloc[::holding_days]

    x = df[
        df["timestamp"].isin(rebalance_dates)
    ].copy()

    x["score"] = score.loc[x.index]

    x["rank"] = (
        x.groupby("timestamp")["score"]
        .rank(
            ascending=False,
            method="first",
        )
    )

    selected = x[x["rank"] <= top_k]

    returns = (
        selected
        .groupby("timestamp")["future_return_5d"]
        .mean()
        .sort_index()
    )

    if len(returns) < 2:
        return None

    equity = (1 + returns).cumprod()

    total_return = equity.iloc[-1] - 1

    years = len(returns) * holding_days / 252

    cagr = (
        equity.iloc[-1] ** (1 / years) - 1
        if years > 0
        else np.nan
    )

    sharpe = (
        np.sqrt(252 / holding_days)
        * returns.mean()
        / returns.std()
        if returns.std() > 0
        else np.nan
    )

    downside = returns[returns < 0].std()

    sortino = (
        np.sqrt(252 / holding_days)
        * returns.mean()
        / downside
        if downside > 0
        else np.nan
    )

    drawdown = (
        equity / equity.cummax()
    ) - 1

    return {
        "periods": len(returns),
        "total_return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "HitRate": (returns > 0).mean(),
        "MaxDrawdown": drawdown.min(),
    }


def main():

    print("=" * 70)
    print("WALK-FORWARD FACTOR VALIDATION")
    print("=" * 70)

    df = pd.read_parquet(INPUT)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values(
        ["timestamp", "symbol"]
    ).reset_index(drop=True)

    scores = build_scores(df)

    # ----------------------------------------------------------
    # IMPORTANT:
    # Model selection period:
    #
    # TRAIN: historical
    # VALID: 2024
    # TEST : 2025+
    #
    # We never use TEST to choose the weights.
    # ----------------------------------------------------------

    train = df[
        df["timestamp"] < "2024-01-01"
    ].copy()

    valid = df[
        (df["timestamp"] >= "2024-01-01")
        & (df["timestamp"] < "2025-01-01")
    ].copy()

    test = df[
        df["timestamp"] >= "2025-01-01"
    ].copy()

    print()
    print("DATA")
    print("-" * 70)
    print("Train:", len(train))
    print("Valid:", len(valid))
    print("Test :", len(test))

    strategies = {
        "RSI": scores["rsi"],
        "COMPOSITE": scores["composite"],
        "20D": scores["r20"],
        "60D": scores["r60"],
        "SMA20": scores["sma20"],
    }

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION — 2024")
    print("=" * 70)

    valid_results = []

    for name, score in strategies.items():

        result = evaluate_period(
            valid,
            score.loc[valid.index],
            top_k=5,
            holding_days=5,
        )

        if result is None:
            continue

        print()
        print(name)

        for k, v in result.items():
            print(f"{k:15s}: {v:.6f}")

        valid_results.append({
            "Strategy": name,
            **result,
        })

    # ----------------------------------------------------------
    # Test
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST — 2025+")
    print("=" * 70)

    test_results = []

    for name, score in strategies.items():

        result = evaluate_period(
            test,
            score.loc[test.index],
            top_k=5,
            holding_days=5,
        )

        if result is None:
            continue

        print()
        print(name)

        for k, v in result.items():
            print(f"{k:15s}: {v:.6f}")

        test_results.append({
            "Strategy": name,
            **result,
        })

    print()
    print("=" * 70)
    print("VALIDATION RANKING")
    print("=" * 70)

    print(
        pd.DataFrame(valid_results)
        .sort_values(
            "Sharpe",
            ascending=False,
        )
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("TEST RANKING")
    print("=" * 70)

    print(
        pd.DataFrame(test_results)
        .sort_values(
            "Sharpe",
            ascending=False,
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
