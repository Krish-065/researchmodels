from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = Path(
    "/workspace/data/features/nse_features.parquet"
)

GRAPH_INPUT = Path(
    "/workspace/data/features/stock_graph.npz"
)

MODEL_OUTPUT = Path(
    "/workspace/data/models/gnn_5d.pt"
)

SEED = 42

LOOKBACK = 60

BATCH_SIZE = 256

EPOCHS = 30

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

PATIENCE = 7

HIDDEN_DIM = 64

DROPOUT = 0.20


FEATURES = [
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


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = SEED):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# GRAPH LAYER
# ============================================================

class GraphConv(nn.Module):
    """
    Simple weighted graph convolution.

    H' = activation(
        W_self H
        +
        W_neighbour A H
    )

    The graph is fixed and constructed only from
    the training period.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.2,
    ):

        super().__init__()

        self.self_linear = nn.Linear(
            in_dim,
            out_dim,
        )

        self.neighbor_linear = nn.Linear(
            in_dim,
            out_dim,
        )

        self.norm = nn.LayerNorm(
            out_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ):

        # x:
        # [batch, nodes, features]

        # adjacency:
        # [nodes, nodes]

        neighbour = torch.matmul(
            adjacency,
            x,
        )

        out = (
            self.self_linear(x)
            +
            self.neighbor_linear(
                neighbour
            )
        )

        out = self.norm(out)

        out = F.relu(out)

        out = self.dropout(out)

        return out


# ============================================================
# GNN MODEL
# ============================================================

class GNNClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
    ):

        super().__init__()

        self.conv1 = GraphConv(
            input_dim,
            hidden_dim,
            dropout,
        )

        self.conv2 = GraphConv(
            hidden_dim,
            hidden_dim,
            dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ):

        x = self.conv1(
            x,
            adjacency,
        )

        x = self.conv2(
            x,
            adjacency,
        )

        logits = self.head(
            x
        ).squeeze(-1)

        return logits


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(
    device: torch.device,
):

    if not GRAPH_INPUT.exists():

        raise FileNotFoundError(
            f"Graph file not found: {GRAPH_INPUT}\n"
            "Run scripts/build_stock_graph.py first."
        )

    graph = np.load(
        GRAPH_INPUT,
        allow_pickle=True,
    )

    symbols = graph[
        "symbols"
    ].astype(str)

    edge_index = graph[
        "edge_index"
    ].astype(np.int64)

    edge_weight = graph[
        "edge_weight"
    ].astype(np.float32)

    n = len(symbols)

    adjacency = np.zeros(
        (n, n),
        dtype=np.float32,
    )

    for src, dst, weight in zip(
        edge_index[0],
        edge_index[1],
        edge_weight,
    ):

        adjacency[
            int(src),
            int(dst),
        ] = float(weight)

    # --------------------------------------------------------
    # Add self loops.
    # --------------------------------------------------------

    for i in range(n):
        adjacency[i, i] = 1.0

    # --------------------------------------------------------
    # Row normalization.
    #
    # Normalize using absolute edge strength so positive
    # and negative correlations remain meaningful.
    # --------------------------------------------------------

    degree = np.sum(
        np.abs(adjacency),
        axis=1,
    )

    degree[
        degree == 0
    ] = 1.0

    adjacency = (
        adjacency
        / degree[:, None]
    )

    adjacency = torch.tensor(
        adjacency,
        dtype=torch.float32,
        device=device,
    )

    return (
        symbols,
        adjacency,
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def load_data():

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
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# BUILD DAILY GRAPH SNAPSHOTS
# ============================================================

def build_daily_data(
    df: pd.DataFrame,
    graph_symbols: np.ndarray,
):

    print()
    print(
        "BUILDING DAILY GRAPH DATA..."
    )

    symbol_to_idx = {
        symbol: i
        for i, symbol in enumerate(
            graph_symbols
        )
    }

    # --------------------------------------------------------
    # Keep only graph symbols.
    # --------------------------------------------------------

    df = df[
        df["symbol"].isin(
            symbol_to_idx
        )
    ].copy()

    df["node_idx"] = df[
        "symbol"
    ].map(
        symbol_to_idx
    )

    df = df.sort_values(
        [
            "timestamp",
            "node_idx",
        ]
    )

    timestamps = sorted(
        df["timestamp"]
        .dropna()
        .unique()
    )

    print(
        "Days:",
        len(timestamps),
    )

    # --------------------------------------------------------
    # Build one graph snapshot per day.
    #
    # X:
    # [days, nodes, features]
    #
    # y:
    # [days, nodes]
    # --------------------------------------------------------

    X_list = []
    y_list = []
    valid_dates = []

    n_nodes = len(
        graph_symbols
    )

    for timestamp in timestamps:

        day = df[
            df["timestamp"]
            == timestamp
        ]

        # Require a complete universe.
        if len(day) != n_nodes:
            continue

        day = day.sort_values(
            "node_idx"
        )

        if (
            day["node_idx"].to_numpy()
            != np.arange(n_nodes)
        ).any():
            continue

        feature_values = (
            day[FEATURES]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
            .to_numpy(
                dtype=np.float32
            )
        )

        targets = (
            day["target_up_5d"]
            .astype(np.float32)
            .to_numpy()
        )

        X_list.append(
            feature_values
        )

        y_list.append(
            targets
        )

        valid_dates.append(
            timestamp
        )

    if not X_list:

        raise RuntimeError(
            "No complete daily graph snapshots found."
        )

    X = np.stack(
        X_list
    )

    y = np.stack(
        y_list
    )

    dates = pd.to_datetime(
        valid_dates,
        utc=True,
    )

    print(
        "Complete graph days:",
        len(X),
    )

    print(
        "X shape:",
        X.shape,
    )

    print(
        "Y shape:",
        y.shape,
    )

    return (
        X,
        y,
        dates,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_features(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
):

    # Calculate statistics using TRAINING DATA ONLY.

    mean = np.nanmean(
        X_train,
        axis=(0, 1),
        keepdims=True,
    )

    std = np.nanstd(
        X_train,
        axis=(0, 1),
        keepdims=True,
    )

    std[
        std < 1e-8
    ] = 1.0

    X_train = (
        X_train - mean
    ) / std

    X_valid = (
        X_valid - mean
    ) / std

    X_test = (
        X_test - mean
    ) / std

    X_train = np.nan_to_num(
        X_train,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_valid = np.nan_to_num(
        X_valid,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_test = np.nan_to_num(
        X_test,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return (
        X_train.astype(
            np.float32
        ),
        X_valid.astype(
            np.float32
        ),
        X_test.astype(
            np.float32
        ),
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probability,
):

    prediction = (
        probability >= 0.5
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true,
            prediction,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            prediction,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probability,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probability,
        ),
        "brier": brier_score_loss(
            y_true,
            probability,
        ),
    }


# ============================================================
# EVALUATE MODEL
# ============================================================

@torch.no_grad()
def predict_dataset(
    model,
    X,
    adjacency,
    device,
):

    model.eval()

    X_tensor = torch.from_numpy(
        X
    ).to(device)

    probabilities = []

    # Process day batches.

    for start in range(
        0,
        len(X_tensor),
        BATCH_SIZE,
    ):

        batch = X_tensor[
            start:
            start + BATCH_SIZE
        ]

        logits = model(
            batch,
            adjacency,
        )

        prob = torch.sigmoid(
            logits
        )

        probabilities.append(
            prob.cpu().numpy()
        )

    return np.concatenate(
        probabilities,
        axis=0,
    )


# ============================================================
# CROSS-SECTIONAL RANKING
# ============================================================

def evaluate_ranking(
    dates,
    symbols,
    y_returns,
    probabilities,
):

    print()
    print(
        "=" * 70
    )
    print(
        "CROSS-SECTIONAL RANKING"
    )
    print(
        "=" * 70
    )

    # y_returns must contain the actual future 5-day returns.

    results = []

    for day_idx, date in enumerate(
        dates
    ):

        scores = probabilities[
            day_idx
        ]

        returns = y_returns[
            day_idx
        ]

        order = np.argsort(
            -scores
        )

        for k in [
            1,
            3,
            5,
            10,
        ]:

            selected = order[
                :min(
                    k,
                    len(order),
                )
            ]

            daily_return = np.mean(
                returns[
                    selected
                ]
            )

            results.append(
                (
                    date,
                    k,
                    daily_return,
                )
            )

    result_df = pd.DataFrame(
        results,
        columns=[
            "timestamp",
            "top_k",
            "return",
        ],
    )

    for k in [
        1,
        3,
        5,
        10,
    ]:

        sub = result_df[
            result_df["top_k"]
            == k
        ]

        daily_returns = (
            sub.groupby(
                "timestamp"
            )["return"]
            .mean()
        )

        mean_return = (
            daily_returns.mean()
        )

        std_return = (
            daily_returns.std()
        )

        sharpe = (
            np.sqrt(
                252 / 5
            )
            * mean_return
            / std_return
            if std_return > 0
            else np.nan
        )

        hit_rate = (
            sub["return"] > 0
        ).mean()

        cumulative = (
            1.0
            + daily_returns
        ).cumprod()

        running_max = (
            cumulative.cummax()
        )

        drawdown = (
            cumulative
            / running_max
            - 1.0
        )

        max_drawdown = (
            drawdown.min()
        )

        print()
        print(
            f"TOP {k:2d}"
        )

        print(
            "Mean 5d return :",
            mean_return,
        )

        print(
            "Std  5d return :",
            std_return,
        )

        print(
            "Approx Sharpe  :",
            sharpe,
        )

        print(
            "Hit rate       :",
            hit_rate,
        )

        print(
            "Max drawdown   :",
            max_drawdown,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    print(
        "=" * 70
    )
    print(
        "PURE PYTORCH GRAPH NEURAL NETWORK"
    )
    print(
        "=" * 70
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device:",
        device,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    # --------------------------------------------------------
    # LOAD GRAPH
    # --------------------------------------------------------

    graph_symbols, adjacency = (
        load_graph(
            device
        )
    )

    print()
    print(
        "Graph nodes:",
        len(graph_symbols),
    )

    print(
        "Adjacency shape:",
        tuple(
            adjacency.shape
        ),
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_data()

    print()
    print(
        "Rows   :",
        len(df),
    )

    print(
        "Symbols:",
        df["symbol"]
        .nunique(),
    )

    print(
        "Features:",
        len(FEATURES),
    )

    # --------------------------------------------------------
    # DAILY GRAPH DATA
    # --------------------------------------------------------

    X, y, dates = (
        build_daily_data(
            df,
            graph_symbols,
        )
    )

    # --------------------------------------------------------
    # TIME SPLIT
    #
    # IMPORTANT:
    # All comparisons use UTC pandas timestamps.
    # No np.datetime64 is used.
    # --------------------------------------------------------

    dates = pd.to_datetime(
        dates,
        utc=True,
    )

    train_start = pd.Timestamp(
        "2010-01-01",
        tz="UTC",
    )

    valid_start = pd.Timestamp(
        "2024-01-01",
        tz="UTC",
    )

    test_start = pd.Timestamp(
        "2025-01-01",
        tz="UTC",
    )

    train_mask = (
        (dates >= train_start)
        & (dates < valid_start)
    )

    valid_mask = (
        (dates >= valid_start)
        & (dates < test_start)
    )

    test_mask = (
        dates >= test_start
    )

    print()
    print(
        "SPLIT"
    )

    print(
        "-" * 70
    )

    print(
        "Train:",
        int(train_mask.sum()),
    )

    print(
        "Valid:",
        int(valid_mask.sum()),
    )

    print(
        "Test :",
        int(test_mask.sum()),
    )

    # --------------------------------------------------------
    # CHECK SPLIT
    # --------------------------------------------------------

    if train_mask.sum() == 0:
        raise RuntimeError(
            "Training split is empty."
        )

    if valid_mask.sum() == 0:
        raise RuntimeError(
            "Validation split is empty."
        )

    if test_mask.sum() == 0:
        raise RuntimeError(
            "Test split is empty."
        )

    # --------------------------------------------------------
    # SPLIT ARRAYS
    # --------------------------------------------------------

    X_train = X[
        train_mask
    ]

    y_train = y[
        train_mask
    ]

    X_valid = X[
        valid_mask
    ]

    y_valid = y[
        valid_mask
    ]

    X_test = X[
        test_mask
    ]

    y_test = y[
        test_mask
    ]

    dates_train = dates[
        train_mask
    ]

    dates_valid = dates[
        valid_mask
    ]

    dates_test = dates[
        test_mask
    ]

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    print()
    print(
        "TARGET"
    )

    print(
        "-" * 70
    )

    print(
        "Train:",
        y_train.mean(),
    )

    print(
        "Valid:",
        y_valid.mean(),
    )

    print(
        "Test :",
        y_test.mean(),
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    (
        X_train,
        X_valid,
        X_test,
        feature_mean,
        feature_std,
    ) = normalize_features(
        X_train,
        X_valid,
        X_test,
    )

    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        torch.from_numpy(
            X_train
        ),
        torch.from_numpy(
            y_train
        ),
    )

    valid_dataset = TensorDataset(
        torch.from_numpy(
            X_valid
        ),
        torch.from_numpy(
            y_valid
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = GNNClassifier(
        input_dim=len(FEATURES),
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "MODEL"
    )

    print(
        "-" * 70
    )

    print(
        model
    )

    print(
        "Parameters:",
        parameter_count,
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    positive_rate = float(
        y_train.mean()
    )

    positive_rate = min(
        max(
            positive_rate,
            1e-4,
        ),
        1 - 1e-4,
    )

    pos_weight = torch.tensor(
        [
            (1.0 - positive_rate)
            / positive_rate
        ],
        dtype=torch.float32,
        device=device,
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "TRAINING"
    )
    print(
        "=" * 70
    )

    best_auc = -np.inf

    best_state = None

    bad_epochs = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        samples_seen = 0

        for xb, yb in train_loader:

            xb = xb.to(
                device,
                non_blocking=True,
            )

            yb = yb.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                xb,
                adjacency,
            )

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size_actual = (
                xb.shape[0]
            )

            running_loss += (
                loss.item()
                * batch_size_actual
            )

            samples_seen += (
                batch_size_actual
            )

        train_loss = (
            running_loss
            / max(
                samples_seen,
                1,
            )
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        valid_prob = predict_dataset(
            model,
            X_valid,
            adjacency,
            device,
        )

        valid_true = (
            y_valid.reshape(-1)
        )

        valid_prob_flat = (
            valid_prob.reshape(-1)
        )

        valid_auc = roc_auc_score(
            valid_true,
            valid_prob_flat,
        )

        valid_pr = (
            average_precision_score(
                valid_true,
                valid_prob_flat,
            )
        )

        scheduler.step(
            valid_auc
        )

        current_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.6f} | "
            f"val_auc={valid_auc:.6f} | "
            f"val_pr={valid_pr:.6f} | "
            f"lr={current_lr:.2e}"
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if valid_auc > best_auc:

            best_auc = valid_auc

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v in model.state_dict().items()
            }

            bad_epochs = 0

        else:

            bad_epochs += 1

            if (
                bad_epochs
                >= PATIENCE
            ):

                print(
                    "Early stopping."
                )

                break

    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "TEST"
    )
    print(
        "=" * 70
    )

    test_prob = predict_dataset(
        model,
        X_test,
        adjacency,
        device,
    )

    test_true = (
        y_test.reshape(-1)
    )

    test_prob_flat = (
        test_prob.reshape(-1)
    )

    metrics = calculate_metrics(
        test_true,
        test_prob_flat,
    )

    print(
        "Accuracy          :",
        metrics["accuracy"],
    )

    print(
        "Balanced accuracy :",
        metrics[
            "balanced_accuracy"
        ],
    )

    print(
        "ROC-AUC           :",
        metrics["roc_auc"],
    )

    print(
        "PR-AUC            :",
        metrics["pr_auc"],
    )

    print(
        "Brier score       :",
        metrics["brier"],
    )

    # --------------------------------------------------------
    # RANKING
    #
    # For ranking we need actual future 5-day returns.
    # --------------------------------------------------------

    test_dates_set = set(
        dates_test
    )

    test_df = df[
        df["timestamp"].isin(
            test_dates_set
        )
    ].copy()

    test_df = test_df[
        test_df["symbol"].isin(
            graph_symbols
        )
    ]

    test_returns = []

    for date in dates_test:

        day = test_df[
            test_df["timestamp"]
            == date
        ].copy()

        day["node_idx"] = (
            day["symbol"]
            .map(
                {
                    symbol: i
                    for i, symbol in enumerate(
                        graph_symbols
                    )
                }
            )
        )

        day = day.sort_values(
            "node_idx"
        )

        if len(day) != len(
            graph_symbols
        ):

            test_returns.append(
                np.zeros(
                    len(
                        graph_symbols
                    ),
                    dtype=np.float32,
                )
            )

        else:

            test_returns.append(
                day[
                    "future_return_5d"
                ]
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float32
                )
            )

    test_returns = np.stack(
        test_returns
    )

    evaluate_ranking(
        dates_test,
        graph_symbols,
        test_returns,
        test_prob,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    MODEL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "features":
                FEATURES,
            "hidden_dim":
                HIDDEN_DIM,
            "dropout":
                DROPOUT,
            "graph_symbols":
                graph_symbols.tolist(),
            "feature_mean":
                feature_mean,
            "feature_std":
                feature_std,
            "best_validation_auc":
                float(best_auc),
            "test_metrics":
                metrics,
        },
        MODEL_OUTPUT,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "MODEL SAVED"
    )
    print(
        "=" * 70
    )

    print(
        MODEL_OUTPUT
    )


if __name__ == "__main__":
    main()
