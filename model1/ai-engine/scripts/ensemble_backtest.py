from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =====================================================================
# PATHS
# =====================================================================

DATA_PATH = Path(
    "/workspace/data/features/nse_features.parquet"
)

LGB_PATH = Path(
    "/workspace/data/models/lightgbm_lambdarank.txt"
)

TCN_PATH = Path(
    "/workspace/data/models/tcn_5d.pt"
)

GNN_PATH = Path(
    "/workspace/data/models/gnn_5d.pt"
)

GRAPH_PATH = Path(
    "/workspace/data/features/stock_graph.npz"
)


# =====================================================================
# FEATURES
# =====================================================================

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


LGB_FEATURES = [
    # Base 17
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

    # Rank 12
    "return_1d_rank",
    "return_5d_rank",
    "return_20d_rank",
    "return_60d_rank",
    "close_sma5_ratio_rank",
    "close_sma20_ratio_rank",
    "close_sma50_ratio_rank",
    "close_sma200_ratio_rank",
    "rsi_14_rank",
    "volatility_20d_rank",
    "volatility_60d_rank",
    "volume_zscore_rank",

    # Z-score 12
    "return_1d_z",
    "return_5d_z",
    "return_20d_z",
    "return_60d_z",
    "close_sma5_ratio_z",
    "close_sma20_ratio_z",
    "close_sma50_ratio_z",
    "close_sma200_ratio_z",
    "rsi_14_z",
    "volatility_20d_z",
    "volatility_60d_z",
    "volume_zscore_z",

    # Market 4
    "market_return_1d",
    "market_return_5d",
    "market_return_20d",
    "market_volatility",

    # Breadth 5
    "breadth_1d",
    "breadth_5d",
    "breadth_sma20",
    "breadth_sma50",
    "breadth_sma200",

    # Relative strength 3
    "relative_return_1d",
    "relative_return_5d",
    "relative_return_20d",

    # Regime 2
    "market_positive_regime",
    "market_high_volatility",
]


# =====================================================================
# DATA
# =====================================================================

def load_data() -> pd.DataFrame:

    print()
    print("LOADING DATA")
    print("-" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    required = [
        "timestamp",
        "symbol",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Dataset missing required columns: "
            + ", ".join(missing)
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df["symbol"] = df["symbol"].astype(str)

    df = (
        df
        .sort_values(
            ["timestamp", "symbol"]
        )
        .reset_index(drop=True)
    )

    print("Rows   :", len(df))
    print(
        "Symbols:",
        df["symbol"].nunique(),
    )
    print(
        "Dates  :",
        df["timestamp"].min(),
        "->",
        df["timestamp"].max(),
    )

    return df


# =====================================================================
# CROSS-SECTIONAL FEATURES
# =====================================================================

def add_cross_sectional_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df = (
        df
        .sort_values(
            ["timestamp", "symbol"]
        )
        .reset_index(drop=True)
    )

    daily = df.groupby(
        "timestamp",
        sort=False,
    )

    # ---------------------------------------------------------------
    # Cross-sectional ranks.
    # ---------------------------------------------------------------

    for col in RANK_FEATURES:

        if col not in df.columns:
            raise RuntimeError(
                f"Missing feature required for ranking: {col}"
            )

        df[f"{col}_rank"] = (
            daily[col]
            .rank(
                pct=True,
                method="average",
            )
        )

    # ---------------------------------------------------------------
    # Cross-sectional z-scores.
    # ---------------------------------------------------------------

    for col in RANK_FEATURES:

        mean = daily[col].transform("mean")
        std = daily[col].transform("std")

        df[f"{col}_z"] = (
            (
                df[col] - mean
            )
            /
            std.replace(0, np.nan)
        )

    # ---------------------------------------------------------------
    # Market state.
    # ---------------------------------------------------------------

    df["market_return_1d"] = (
        daily["return_1d"]
        .transform("mean")
    )

    df["market_return_5d"] = (
        daily["return_5d"]
        .transform("mean")
    )

    df["market_return_20d"] = (
        daily["return_20d"]
        .transform("mean")
    )

    df["market_volatility"] = (
        daily["volatility_20d"]
        .transform("mean")
    )

    # ---------------------------------------------------------------
    # Breadth.
    # ---------------------------------------------------------------

    df["breadth_1d"] = (
        daily["return_1d"]
        .transform(
            lambda x: (x > 0).mean()
        )
    )

    df["breadth_5d"] = (
        daily["return_5d"]
        .transform(
            lambda x: (x > 0).mean()
        )
    )

    df["breadth_sma20"] = (
        daily["close_sma20_ratio"]
        .transform(
            lambda x: (x > 0).mean()
        )
    )

    df["breadth_sma50"] = (
        daily["close_sma50_ratio"]
        .transform(
            lambda x: (x > 0).mean()
        )
    )

    df["breadth_sma200"] = (
        daily["close_sma200_ratio"]
        .transform(
            lambda x: (x > 0).mean()
        )
    )

    # ---------------------------------------------------------------
    # Relative returns.
    # ---------------------------------------------------------------

    df["relative_return_1d"] = (
        df["return_1d"]
        - df["market_return_1d"]
    )

    df["relative_return_5d"] = (
        df["return_5d"]
        - df["market_return_5d"]
    )

    df["relative_return_20d"] = (
        df["return_20d"]
        - df["market_return_20d"]
    )

    # ---------------------------------------------------------------
    # Historical volatility regime.
    #
    # IMPORTANT:
    # The median is calculated from historical market-volatility
    # observations. Today's volatility is never compared against
    # a same-day cross-sectional median.
    # ---------------------------------------------------------------

    market_vol = (
        df[
            [
                "timestamp",
                "market_volatility",
            ]
        ]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .copy()
    )

    market_vol[
        "market_vol_median_60d"
    ] = (
        market_vol[
            "market_volatility"
        ]
        .rolling(
            60,
            min_periods=20,
        )
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
        validate="many_to_one",
    )

    df["market_high_volatility"] = (
        df["market_volatility"]
        >
        df["market_vol_median_60d"]
    ).astype("int8")

    df["market_positive_regime"] = (
        df["market_return_20d"] > 0
    ).astype("int8")

    return df


# =====================================================================
# FEATURE CLEANING
# =====================================================================

def clean_features(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    missing = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required features: "
            + ", ".join(missing)
        )

    return (
        df[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )


# =====================================================================
# LIGHTGBM
# =====================================================================

def load_lightgbm():

    import lightgbm as lgb

    print()
    print("LOADING LIGHTGBM")
    print("-" * 70)

    if not LGB_PATH.exists():
        raise FileNotFoundError(
            f"LightGBM model not found: {LGB_PATH}"
        )

    model = lgb.Booster(
        model_file=str(LGB_PATH)
    )

    checkpoint_features = (
        model.feature_name()
    )

    print(
        "LightGBM features:",
        len(checkpoint_features),
    )

    if list(checkpoint_features) != LGB_FEATURES:

        print(
            "WARNING: checkpoint feature schema "
            "differs from expected schema."
        )

        print(
            "Using feature order from checkpoint."
        )

    else:

        print(
            "Feature schema verified: 55 features."
        )

    return model


def predict_lightgbm(
    model,
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("GENERATING LIGHTGBM SCORES...")
    print("-" * 70)

    features = model.feature_name()

    x = clean_features(
        df,
        features,
    )

    pred = model.predict(x)

    pred = np.asarray(
        pred,
        dtype=np.float64,
    ).reshape(-1)

    if len(pred) != len(df):
        raise RuntimeError(
            "LightGBM prediction length mismatch."
        )

    out = df.copy()

    out["lightgbm_score"] = pred

    print(
        "Valid LightGBM scores:",
        int(
            np.isfinite(pred).sum()
        ),
        "/",
        len(pred),
    )

    print(
        "Score mean:",
        float(np.nanmean(pred)),
    )

    print(
        "Score std :",
        float(np.nanstd(pred)),
    )

    return out


# =====================================================================
# FACTOR SCORE
# =====================================================================

def add_factor_score(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    factors = {
        "return_20d": 0.30,
        "return_60d": 0.15,
        "close_sma20_ratio": 0.20,
        "rsi_14": 0.15,
        "close_sma50_ratio": 0.10,
        "volume_zscore": 0.10,
    }

    total = np.zeros(
        len(df),
        dtype=np.float64,
    )

    weight_sum = 0.0

    for col, weight in factors.items():

        if col not in df.columns:
            continue

        rank = (
            df.groupby("timestamp")[col]
            .rank(
                pct=True,
                method="average",
            )
        )

        values = rank.to_numpy(
            dtype=np.float64
        )

        values = np.where(
            np.isfinite(values),
            values,
            0.5,
        )

        total += (
            weight * values
        )

        weight_sum += weight

    if weight_sum > 0:

        df["factor_score"] = (
            total / weight_sum
        )

    else:

        df["factor_score"] = np.nan

    return df


# =====================================================================
# MOMENTUM
# =====================================================================

def add_momentum_score(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    ranks = []

    for col in [
        "return_20d",
        "return_60d",
    ]:

        if col not in df.columns:
            continue

        r = (
            df.groupby("timestamp")[col]
            .rank(
                pct=True,
                method="average",
            )
        )

        ranks.append(
            r.fillna(0.5)
        )

    if ranks:

        df["momentum_score"] = (
            pd.concat(
                ranks,
                axis=1,
            )
            .mean(axis=1)
        )

    else:

        df["momentum_score"] = np.nan

    return df


# =====================================================================
# TCN
# =====================================================================

def predict_tcn(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("GENERATING TCN SCORES...")
    print("-" * 70)

    try:
        import torch
        import torch.nn as nn
    except Exception as exc:

        print(
            "TCN unavailable:",
            repr(exc),
        )

        out = df.copy()
        out["tcn_score"] = np.nan
        return out

    if not TCN_PATH.exists():

        print(
            "TCN checkpoint missing:",
            TCN_PATH,
        )

        out = df.copy()
        out["tcn_score"] = np.nan
        return out

    try:

        checkpoint = torch.load(
            TCN_PATH,
            map_location="cpu",
            weights_only=False,
        )

        print(
            "TCN checkpoint loaded."
        )

    except Exception as exc:

        print(
            "Could not load TCN:",
            repr(exc),
        )

        out = df.copy()
        out["tcn_score"] = np.nan
        return out

    class Chomp1d(nn.Module):

        def __init__(self, chomp_size):

            super().__init__()

            self.chomp_size = chomp_size

        def forward(self, x):

            if self.chomp_size == 0:
                return x

            return x[
                :,
                :,
                :-self.chomp_size
            ].contiguous()

    class TemporalBlock(nn.Module):

        def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            dilation,
            dropout,
        ):

            super().__init__()

            padding = (
                kernel_size - 1
            ) * dilation

            self.conv1 = nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )

            self.chomp1 = Chomp1d(
                padding
            )

            self.relu1 = nn.ReLU()

            self.dropout1 = nn.Dropout(
                dropout
            )

            self.conv2 = nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )

            self.chomp2 = Chomp1d(
                padding
            )

            self.relu2 = nn.ReLU()

            self.dropout2 = nn.Dropout(
                dropout
            )

            self.downsample = (
                nn.Conv1d(
                    n_inputs,
                    n_outputs,
                    1,
                )
                if n_inputs != n_outputs
                else None
            )

            self.final_relu = nn.ReLU()

        def forward(self, x):

            out = self.conv1(x)
            out = self.chomp1(out)
            out = self.relu1(out)
            out = self.dropout1(out)

            out = self.conv2(out)
            out = self.chomp2(out)
            out = self.relu2(out)
            out = self.dropout2(out)

            residual = x

            if self.downsample is not None:
                residual = self.downsample(x)

            return self.final_relu(
                out + residual
            )

    class TCNClassifier(nn.Module):

        def __init__(
            self,
            num_features,
        ):

            super().__init__()

            channels = [
                64,
                64,
                128,
                128,
            ]

            blocks = []

            in_channels = num_features

            for i, out_channels in enumerate(
                channels
            ):

                blocks.append(
                    TemporalBlock(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        dilation=2 ** i,
                        dropout=0.2,
                    )
                )

                in_channels = out_channels

            self.tcn = nn.Sequential(
                *blocks
            )

            self.head = nn.Sequential(
                nn.Linear(
                    128,
                    64,
                ),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(
                    64,
                    1,
                ),
            )

        def forward(self, x):

            z = self.tcn(x)

            z = z[:, :, -1]

            return self.head(
                z
            ).squeeze(-1)

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state = checkpoint[
                "state_dict"
            ]

        else:

            state = checkpoint

    else:

        state = checkpoint

    model = TCNClassifier(
        len(BASE_FEATURES)
    )

    try:

        model.load_state_dict(
            state,
            strict=True,
        )

    except Exception as exc:

        print(
            "TCN state loading failed:",
            repr(exc),
        )

        out = df.copy()
        out["tcn_score"] = np.nan
        return out

    model.eval()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model.to(device)

    print(
        "TCN device:",
        device,
    )

    feature_df = clean_features(
        df,
        BASE_FEATURES,
    ).copy()

    feature_df["timestamp"] = (
        df["timestamp"].to_numpy()
    )

    feature_df["symbol"] = (
        df["symbol"].to_numpy()
    )

    feature_df["_row_id"] = np.arange(
        len(df)
    )

    scores = np.full(
        len(df),
        np.nan,
        dtype=np.float64,
    )

    lookback = 60
    batch_size = 2048

    with torch.no_grad():

        for symbol, group in feature_df.groupby(
            "symbol",
            sort=False,
        ):

            group = group.sort_values(
                "timestamp"
            )

            rows = group[
                "_row_id"
            ].to_numpy()

            values = group[
                BASE_FEATURES
            ].to_numpy(
                dtype=np.float32
            )

            if len(values) < lookback:
                continue

            sequences = []
            target_rows = []

            for i in range(
                lookback,
                len(values) + 1,
            ):

                sequences.append(
                    values[
                        i - lookback:i
                    ].T
                )

                target_rows.append(
                    rows[i - 1]
                )

            if not sequences:
                continue

            batch = np.stack(
                sequences
            )

            predictions = []

            for start in range(
                0,
                len(batch),
                batch_size,
            ):

                part = torch.tensor(
                    batch[
                        start:
                        start + batch_size
                    ],
                    dtype=torch.float32,
                    device=device,
                )

                logits = model(part)

                probabilities = (
                    torch.sigmoid(logits)
                    .detach()
                    .cpu()
                    .numpy()
                )

                predictions.append(
                    probabilities
                )

            predictions = np.concatenate(
                predictions
            )

            scores[
                np.asarray(target_rows)
            ] = predictions

    out = df.copy()

    out["tcn_score"] = scores

    print(
        "Valid TCN scores:",
        int(
            np.isfinite(scores).sum()
        ),
        "/",
        len(scores),
    )

    return out


# =====================================================================
# GNN
# =====================================================================

def predict_gnn(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("GENERATING GNN SCORES...")
    print("-" * 70)

    try:
        import torch
        import torch.nn as nn
    except Exception as exc:

        print(
            "GNN unavailable:",
            repr(exc),
        )

        out = df.copy()
        out["gnn_score"] = np.nan
        return out

    if not GNN_PATH.exists():

        print(
            "GNN checkpoint missing:",
            GNN_PATH,
        )

        out = df.copy()
        out["gnn_score"] = np.nan
        return out

    if not GRAPH_PATH.exists():

        print(
            "Graph file missing:",
            GRAPH_PATH,
        )

        out = df.copy()
        out["gnn_score"] = np.nan
        return out

    try:

        checkpoint = torch.load(
            GNN_PATH,
            map_location="cpu",
            weights_only=False,
        )

        graph = np.load(
            GRAPH_PATH
        )

    except Exception as exc:

        print(
            "Could not load GNN resources:",
            repr(exc),
        )

        out = df.copy()
        out["gnn_score"] = np.nan
        return out

    symbols = [
        str(x)
        for x in graph["symbols"]
    ]

    edge_index = graph[
        "edge_index"
    ]

    edge_weight = graph[
        "edge_weight"
    ]

    num_nodes = len(symbols)

    adjacency = np.zeros(
        (
            num_nodes,
            num_nodes,
        ),
        dtype=np.float32,
    )

    for i in range(
        edge_index.shape[1]
    ):

        src = int(
            edge_index[0, i]
        )

        dst = int(
            edge_index[1, i]
        )

        w = float(
            edge_weight[i]
        )

        adjacency[
            dst,
            src
        ] = w

    adjacency += np.eye(
        num_nodes,
        dtype=np.float32,
    )

    row_sum = adjacency.sum(
        axis=1,
        keepdims=True,
    )

    row_sum[
        row_sum == 0
    ] = 1.0

    adjacency = (
        adjacency
        / row_sum
    )

    class GraphConv(nn.Module):

        def __init__(
            self,
            in_features,
            out_features,
            dropout=0.2,
        ):

            super().__init__()

            self.self_linear = nn.Linear(
                in_features,
                out_features,
            )

            self.neighbor_linear = nn.Linear(
                in_features,
                out_features,
            )

            self.norm = nn.LayerNorm(
                out_features
            )

            self.dropout = nn.Dropout(
                dropout
            )

        def forward(
            self,
            x,
            adj,
        ):

            self_part = (
                self.self_linear(x)
            )

            neighbor = torch.matmul(
                adj,
                x,
            )

            neighbor_part = (
                self.neighbor_linear(
                    neighbor
                )
            )

            out = (
                self_part
                + neighbor_part
            )

            out = self.norm(out)
            out = torch.relu(out)
            out = self.dropout(out)

            return out

    class GNNClassifier(nn.Module):

        def __init__(
            self,
            num_features,
        ):

            super().__init__()

            self.conv1 = GraphConv(
                num_features,
                64,
                0.2,
            )

            self.conv2 = GraphConv(
                64,
                64,
                0.2,
            )

            self.head = nn.Sequential(
                nn.Linear(
                    64,
                    32,
                ),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(
                    32,
                    1,
                ),
            )

        def forward(
            self,
            x,
            adj,
        ):

            x = self.conv1(
                x,
                adj,
            )

            x = self.conv2(
                x,
                adj,
            )

            return self.head(
                x
            ).squeeze(-1)

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state = checkpoint[
                "state_dict"
            ]

        else:

            state = checkpoint

    else:

        state = checkpoint

    model = GNNClassifier(
        len(BASE_FEATURES)
    )

    try:

        model.load_state_dict(
            state,
            strict=True,
        )

    except Exception as exc:

        print(
            "GNN state loading failed:",
            repr(exc),
        )

        out = df.copy()
        out["gnn_score"] = np.nan
        return out

    model.eval()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model.to(device)

    print(
        "GNN device:",
        device,
    )

    adj = torch.tensor(
        adjacency,
        dtype=torch.float32,
        device=device,
    )

    feature_data = clean_features(
        df,
        BASE_FEATURES,
    ).copy()

    feature_data["timestamp"] = (
        df["timestamp"].to_numpy()
    )

    feature_data["symbol"] = (
        df["symbol"].to_numpy()
    )

    feature_data["_row_id"] = np.arange(
        len(df)
    )

    scores = np.full(
        len(df),
        np.nan,
        dtype=np.float64,
    )

    symbol_to_idx = {
        symbol: i
        for i, symbol in enumerate(
            symbols
        )
    }

    with torch.no_grad():

        for timestamp, group in feature_data.groupby(
            "timestamp",
            sort=True,
        ):

            if len(group) != num_nodes:
                continue

            node_features = np.zeros(
                (
                    num_nodes,
                    len(BASE_FEATURES),
                ),
                dtype=np.float32,
            )

            row_ids = np.zeros(
                num_nodes,
                dtype=np.int64,
            )

            valid = True
            seen = set()

            for _, row in group.iterrows():

                symbol = str(
                    row["symbol"]
                )

                if symbol not in symbol_to_idx:
                    valid = False
                    break

                idx = symbol_to_idx[
                    symbol
                ]

                if idx in seen:
                    valid = False
                    break

                seen.add(idx)

                node_features[
                    idx
                ] = row[
                    BASE_FEATURES
                ].to_numpy(
                    dtype=np.float32
                )

                row_ids[
                    idx
                ] = int(
                    row["_row_id"]
                )

            if not valid:
                continue

            if len(seen) != num_nodes:
                continue

            x = torch.tensor(
                node_features,
                dtype=torch.float32,
                device=device,
            )

            logits = model(
                x,
                adj,
            )

            probabilities = (
                torch.sigmoid(logits)
                .detach()
                .cpu()
                .numpy()
            )

            scores[
                row_ids
            ] = probabilities

    out = df.copy()

    out["gnn_score"] = scores

    print(
        "Valid GNN scores:",
        int(
            np.isfinite(scores).sum()
        ),
        "/",
        len(scores),
    )

    return out


# =====================================================================
# ENSEMBLE
# =====================================================================

def build_ensemble(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    score_columns = [
        "lightgbm_score",
        "factor_score",
        "momentum_score",
        "tcn_score",
        "gnn_score",
    ]

    for col in score_columns:

        if col not in df.columns:
            df[col] = np.nan

    # ---------------------------------------------------------------
    # Convert each model's score into a cross-sectional percentile.
    #
    # Higher raw score = better percentile.
    #
    # IMPORTANT:
    # Missing model predictions stay missing.
    # ---------------------------------------------------------------

    for col in score_columns:

        df[
            f"{col}_rank"
        ] = (
            df.groupby("timestamp")[col]
            .rank(
                ascending=False,
                pct=True,
                method="average",
            )
        )

    # ---------------------------------------------------------------
    # Baseline ensemble.
    #
    # These are deliberately unchanged from the previous experiment.
    # We first want a clean, reproducible baseline before optimizing
    # weights.
    # ---------------------------------------------------------------

    components = [
        (
            "lightgbm_score_rank",
            0.50,
        ),
        (
            "factor_score_rank",
            0.25,
        ),
        (
            "momentum_score_rank",
            0.15,
        ),
        (
            "tcn_score_rank",
            0.05,
        ),
        (
            "gnn_score_rank",
            0.05,
        ),
    ]

    total = np.zeros(
        len(df),
        dtype=np.float64,
    )

    weights = np.zeros(
        len(df),
        dtype=np.float64,
    )

    for col, weight in components:

        values = df[col].to_numpy(
            dtype=np.float64
        )

        valid = np.isfinite(values)

        total[valid] += (
            weight
            * values[valid]
        )

        weights[valid] += weight

    df["ensemble_score"] = np.divide(
        total,
        weights,
        out=np.full(
            len(df),
            np.nan,
            dtype=np.float64,
        ),
        where=weights > 0,
    )

    return df


# =====================================================================
# BACKTEST
# =====================================================================

def backtest_top_k(
    df: pd.DataFrame,
    score_col: str,
    top_k: int,
    holding_days: int = 5,
):
    """
    Non-overlapping 5-trading-day backtest.

    At rebalance date t:
        - use score at t
        - select top-k stocks
        - use future_return_5d from t

    Rebalance dates are t, t+5, t+10, ...

    This means each reported portfolio period corresponds to a
    distinct five-trading-day interval.
    """

    if score_col not in df.columns:
        return None

    if "future_return_5d" not in df.columns:
        raise RuntimeError(
            "future_return_5d is required."
        )

    work = df.copy()

    work = (
        work
        .sort_values(
            ["timestamp", "symbol"]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Only dates with at least one usable observation can be
    # considered. We then step through actual trading dates.
    # ---------------------------------------------------------------

    dates = (
        work["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    rows = []

    date_index = 0

    while date_index < len(dates):

        timestamp = dates.iloc[
            date_index
        ]

        group = work[
            work["timestamp"] == timestamp
        ].copy()

        group[score_col] = pd.to_numeric(
            group[score_col],
            errors="coerce",
        )

        group["future_return_5d"] = (
            pd.to_numeric(
                group["future_return_5d"],
                errors="coerce",
            )
        )

        group = group.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        group = group.dropna(
            subset=[
                score_col,
                "future_return_5d",
            ]
        )

        if len(group) >= top_k:

            selected = (
                group
                .sort_values(
                    score_col,
                    ascending=False,
                    kind="mergesort",
                )
                .head(top_k)
            )

            portfolio_return = float(
                selected[
                    "future_return_5d"
                ].mean()
            )

            rows.append(
                {
                    "timestamp": timestamp,
                    "return": portfolio_return,
                    "n_selected": len(selected),
                }
            )

        # -----------------------------------------------------------
        # Move exactly five trading sessions forward.
        # -----------------------------------------------------------

        date_index += holding_days

    if not rows:
        return None

    result = pd.DataFrame(rows)

    result = (
        result
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    returns = result[
        "return"
    ].to_numpy(
        dtype=np.float64
    )

    returns = returns[
        np.isfinite(returns)
    ]

    if len(returns) == 0:
        return None

    # ---------------------------------------------------------------
    # Guard against impossible losses.
    # ---------------------------------------------------------------

    returns = np.clip(
        returns,
        -0.999999,
        None,
    )

    # ---------------------------------------------------------------
    # Equity.
    # ---------------------------------------------------------------

    equity = np.cumprod(
        1.0 + returns
    )

    total_return = (
        equity[-1] - 1.0
    )

    periods = len(returns)

    years = (
        periods
        * holding_days
        / 252.0
    )

    if (
        years > 0
        and equity[-1] > 0
    ):

        cagr = (
            equity[-1]
            ** (1.0 / years)
            - 1.0
        )

    else:

        cagr = np.nan

    # ---------------------------------------------------------------
    # Sharpe.
    # ---------------------------------------------------------------

    if len(returns) > 1:

        std = float(
            np.std(
                returns,
                ddof=1,
            )
        )

    else:

        std = 0.0

    if std > 0:

        sharpe = (
            float(
                np.mean(returns)
            )
            / std
            * np.sqrt(
                252.0 / holding_days
            )
        )

    else:

        sharpe = np.nan

    # ---------------------------------------------------------------
    # Sortino.
    # ---------------------------------------------------------------

    downside = returns[
        returns < 0
    ]

    if len(downside) > 1:

        downside_std = float(
            np.std(
                downside,
                ddof=1,
            )
        )

    else:

        downside_std = 0.0

    if downside_std > 0:

        sortino = (
            float(
                np.mean(returns)
            )
            / downside_std
            * np.sqrt(
                252.0 / holding_days
            )
        )

    else:

        sortino = np.nan

    # ---------------------------------------------------------------
    # Hit rate.
    # ---------------------------------------------------------------

    hit_rate = float(
        np.mean(
            returns > 0
        )
    )

    # ---------------------------------------------------------------
    # Maximum drawdown.
    # ---------------------------------------------------------------

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    max_drawdown = float(
        np.min(drawdown)
    )

    return {
        "total_return": float(
            total_return
        ),
        "CAGR": float(
            cagr
        ),
        "Sharpe": float(
            sharpe
        ),
        "Sortino": float(
            sortino
        ),
        "HitRate": hit_rate,
        "MaxDrawdown": max_drawdown,
        "periods": periods,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print("ENSEMBLE BACKTEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------------

    df = load_data()

    # ---------------------------------------------------------------
    # AUDIT TARGET COLUMNS BEFORE MODEL INFERENCE.
    # ---------------------------------------------------------------

    future_columns = [
        c
        for c in df.columns
        if (
            "future" in c.lower()
            or "target" in c.lower()
            or "label" in c.lower()
        )
    ]

    print()
    print("TARGET / FUTURE COLUMNS")
    print("-" * 70)

    for col in future_columns:
        print(col)

    forbidden_model_inputs = set(
        future_columns
    )

    model_input_overlap = (
        forbidden_model_inputs
        & set(BASE_FEATURES)
    )

    if model_input_overlap:

        raise RuntimeError(
            "Future/target column found in BASE_FEATURES: "
            + ", ".join(
                sorted(model_input_overlap)
            )
        )

    # ---------------------------------------------------------------
    # CROSS-SECTIONAL FEATURES
    # ---------------------------------------------------------------

    print()
    print(
        "BUILDING CROSS-SECTIONAL FEATURES..."
    )
    print("-" * 70)

    df = add_cross_sectional_features(
        df
    )

    print(
        "Cross-sectional features ready."
    )

    # ---------------------------------------------------------------
    # TEST PERIOD
    # ---------------------------------------------------------------

    test_start = pd.Timestamp(
        "2025-01-01",
        tz="UTC",
    )

    test = df[
        df["timestamp"]
        >= test_start
    ].copy()

    print()
    print("TEST")
    print("-" * 70)

    print(
        "Rows   :",
        len(test),
    )

    print(
        "Days   :",
        test["timestamp"].nunique(),
    )

    print(
        "Symbols:",
        test["symbol"].nunique(),
    )

    # ---------------------------------------------------------------
    # FUTURE RETURN
    #
    # Prefer the already-generated target from the feature dataset.
    # If it does not exist, reconstruct it from close.
    # ---------------------------------------------------------------

    if "future_return_5d" in df.columns:

        print()
        print(
            "Using existing future_return_5d."
        )

    else:

        print()
        print(
            "Building future_return_5d..."
        )

        full = (
            df
            .sort_values(
                ["symbol", "timestamp"]
            )
            .copy()
        )

        full["future_return_5d"] = (
            full.groupby("symbol")[
                "close"
            ]
            .shift(-5)
            / full["close"]
            - 1.0
        )

        df = full

        test = df[
            df["timestamp"]
            >= test_start
        ].copy()

    # ---------------------------------------------------------------
    # IMPORTANT TARGET SANITY CHECK
    #
    # We should not have an infinite target.
    # Rows near the dataset end may have missing future returns.
    # ---------------------------------------------------------------

    test["future_return_5d"] = pd.to_numeric(
        test["future_return_5d"],
        errors="coerce",
    )

    target_valid = np.isfinite(
        test["future_return_5d"]
        .to_numpy(
            dtype=np.float64
        )
    )

    print()
    print(
        "Valid future_return_5d:",
        int(target_valid.sum()),
        "/",
        len(test),
    )

    # ---------------------------------------------------------------
    # LIGHTGBM
    # ---------------------------------------------------------------

    lgb_model = load_lightgbm()

    test = predict_lightgbm(
        lgb_model,
        test,
    )

    # ---------------------------------------------------------------
    # FACTORS
    # ---------------------------------------------------------------

    print()
    print(
        "GENERATING FACTOR SCORES..."
    )
    print("-" * 70)

    test = add_factor_score(
        test
    )

    test = add_momentum_score(
        test
    )

    # ---------------------------------------------------------------
    # TCN
    #
    # Generate on FULL DATA so each test-date sequence has its
    # historical 60-session context.
    # ---------------------------------------------------------------

    tcn_all = predict_tcn(
        df
    )

    tcn_lookup = tcn_all[
        [
            "timestamp",
            "symbol",
            "tcn_score",
        ]
    ].copy()

    test = test.drop(
        columns=[
            "tcn_score",
        ],
        errors="ignore",
    )

    test = test.merge(
        tcn_lookup,
        on=[
            "timestamp",
            "symbol",
        ],
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------
    # GNN
    #
    # Generate on FULL DATA because the graph is cross-sectional.
    # ---------------------------------------------------------------

    gnn_all = predict_gnn(
        df
    )

    gnn_lookup = gnn_all[
        [
            "timestamp",
            "symbol",
            "gnn_score",
        ]
    ].copy()

    test = test.drop(
        columns=[
            "gnn_score",
        ],
        errors="ignore",
    )

    test = test.merge(
        gnn_lookup,
        on=[
            "timestamp",
            "symbol",
        ],
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------
    # BUILD ENSEMBLE
    # ---------------------------------------------------------------

    test = build_ensemble(
        test
    )

    # ---------------------------------------------------------------
    # SCORE COVERAGE
    # ---------------------------------------------------------------

    print()
    print("SCORE COVERAGE")
    print("-" * 70)

    for col in [
        "lightgbm_score",
        "factor_score",
        "momentum_score",
        "tcn_score",
        "gnn_score",
        "ensemble_score",
    ]:

        values = test[
            col
        ].to_numpy(
            dtype=float
        )

        valid = int(
            np.isfinite(values).sum()
        )

        print(
            f"{col:25s}: "
            f"{valid:6d} / {len(test)}"
        )

    # ---------------------------------------------------------------
    # BACKTEST INPUT
    # ---------------------------------------------------------------

    print()
    print("BACKTEST INPUT")
    print("-" * 70)

    print(
        "Rows       :",
        len(test),
    )

    print(
        "Dates      :",
        test["timestamp"].nunique(),
    )

    print(
        "Symbols    :",
        test["symbol"].nunique(),
    )

    print(
        "Future 5d  :",
        int(
            np.isfinite(
                test[
                    "future_return_5d"
                ].to_numpy(
                    dtype=float
                )
            ).sum()
        ),
        "/",
        len(test),
    )

    # ---------------------------------------------------------------
    # BACKTEST
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "NON-OVERLAPPING PORTFOLIO RESULTS"
    )
    print("=" * 70)

    results = []

    strategies = [
        (
            "LIGHTGBM",
            "lightgbm_score",
        ),
        (
            "FACTOR",
            "factor_score",
        ),
        (
            "MOMENTUM",
            "momentum_score",
        ),
        (
            "TCN",
            "tcn_score",
        ),
        (
            "GNN",
            "gnn_score",
        ),
        (
            "ENSEMBLE",
            "ensemble_score",
        ),
    ]

    for name, score_col in strategies:

        print()
        print(name)
        print("-" * 70)

        for top_k in [
            1,
            3,
            5,
            10,
        ]:

            metrics = backtest_top_k(
                test,
                score_col,
                top_k,
                holding_days=5,
            )

            if metrics is None:

                print(
                    f"TOP {top_k}: "
                    "NO VALID DATA"
                )

                continue

            print(
                f"TOP {top_k}"
            )

            print(
                f"total_return : "
                f"{metrics['total_return']:.6f}"
            )

            print(
                f"CAGR         : "
                f"{metrics['CAGR']:.6f}"
            )

            print(
                f"Sharpe       : "
                f"{metrics['Sharpe']:.6f}"
            )

            print(
                f"Sortino      : "
                f"{metrics['Sortino']:.6f}"
            )

            print(
                f"HitRate      : "
                f"{metrics['HitRate']:.6f}"
            )

            print(
                f"MaxDrawdown  : "
                f"{metrics['MaxDrawdown']:.6f}"
            )

            print(
                f"periods      : "
                f"{metrics['periods']}"
            )

            results.append(
                {
                    "Strategy": name,
                    "TopK": top_k,
                    **metrics,
                }
            )

    # ---------------------------------------------------------------
    # FINAL TABLE
    # ---------------------------------------------------------------

    if results:

        summary = pd.DataFrame(
            results
        )

        print()
        print("=" * 70)
        print(
            "FINAL SUMMARY"
        )
        print("=" * 70)

        print(
            summary.to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print(
        "ENSEMBLE BACKTEST COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
