from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT = Path(
    "/workspace/data/features/nse_features.parquet"
)

OUTPUT = Path(
    "/workspace/data/features/stock_graph.npz"
)

CORRELATION_WINDOW = 60
TOP_K = 8

FEATURE = "return_1d"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILD STOCK CORRELATION GRAPH")
    print("=" * 70)

    df = pd.read_parquet(
        INPUT
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values(
        [
            "timestamp",
            "symbol",
        ]
    )

    print()
    print(
        "Rows   :",
        len(df),
    )

    symbols = sorted(
        df["symbol"].dropna().unique()
    )

    print(
        "Symbols:",
        len(symbols),
    )

    # --------------------------------------------------------
    # BUILD RETURN MATRIX
    # --------------------------------------------------------

    returns = (
        df.pivot(
            index="timestamp",
            columns="symbol",
            values=FEATURE,
        )
        .reindex(
            columns=symbols
        )
    )

    print(
        "Return matrix:",
        returns.shape,
    )

    # --------------------------------------------------------
    # USE TRAIN PERIOD ONLY
    #
    # IMPORTANT:
    # The graph must not use 2025+ test information.
    # --------------------------------------------------------

    train_returns = returns[
        returns.index < pd.Timestamp(
            "2024-01-01",
            tz="UTC",
        )
    ]

    print()
    print(
        "Training-period rows:",
        len(train_returns),
    )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    corr = train_returns.corr(
        min_periods=100,
    )

    corr = corr.reindex(
        index=symbols,
        columns=symbols,
    )

    corr_values = (
        corr
        .fillna(0.0)
        .to_numpy(
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # BUILD TOP-K GRAPH
    #
    # For every stock:
    # connect to the TOP_K most correlated
    # other stocks by absolute correlation.
    # --------------------------------------------------------

    edge_src = []
    edge_dst = []
    edge_weight = []

    n = len(symbols)

    for i in range(n):

        scores = corr_values[i].copy()

        # Never connect stock to itself.
        scores[i] = 0.0

        # Rank by absolute correlation.
        order = np.argsort(
            -np.abs(scores)
        )

        selected = order[
            :TOP_K
        ]

        for j in selected:

            if i == j:
                continue

            weight = float(
                scores[j]
            )

            # Ignore zero relationships.
            if not np.isfinite(weight):
                continue

            if weight == 0:
                continue

            edge_src.append(i)
            edge_dst.append(int(j))
            edge_weight.append(weight)

    edge_index = np.asarray(
        [
            edge_src,
            edge_dst,
        ],
        dtype=np.int64,
    )

    edge_weight = np.asarray(
        edge_weight,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # SYMMETRIZE
    #
    # If A -> B exists, also create B -> A.
    # --------------------------------------------------------

    existing = {
        (
            int(a),
            int(b),
        )
        for a, b in zip(
            edge_src,
            edge_dst,
        )
    }

    extra_src = []
    extra_dst = []
    extra_weight = []

    for a, b, w in zip(
        edge_src,
        edge_dst,
        edge_weight,
    ):

        if (
            b,
            a,
        ) not in existing:

            extra_src.append(b)
            extra_dst.append(a)
            extra_weight.append(w)

    if extra_src:

        edge_index = np.concatenate(
            [
                edge_index,
                np.asarray(
                    [
                        extra_src,
                        extra_dst,
                    ],
                    dtype=np.int64,
                ),
            ],
            axis=1,
        )

        edge_weight = np.concatenate(
            [
                edge_weight,
                np.asarray(
                    extra_weight,
                    dtype=np.float32,
                ),
            ]
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        OUTPUT,
        symbols=np.asarray(
            symbols
        ),
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "GRAPH"
    )

    print(
        "=" * 70
    )

    print(
        "Nodes:",
        n,
    )

    print(
        "Edges:",
        edge_index.shape[1],
    )

    print(
        "Average degree:",
        edge_index.shape[1] / n,
    )

    print(
        "Weight min:",
        edge_weight.min(),
    )

    print(
        "Weight max:",
        edge_weight.max(),
    )

    print(
        "Mean |weight|:",
        np.mean(
            np.abs(
                edge_weight
            )
        ),
    )

    print()
    print(
        "MODEL GRAPH SAVED:"
    )

    print(
        OUTPUT
    )


if __name__ == "__main__":
    main()
