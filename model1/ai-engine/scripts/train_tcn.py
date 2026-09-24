from __future__ import annotations

from pathlib import Path

import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


# ============================================================
# CONFIG
# ============================================================

INPUT = Path("/workspace/data/features/nse_features.parquet")
OUTPUT = Path("/workspace/data/models/tcn_5d.pt")

LOOKBACK = 60

BATCH_SIZE = 512
EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 7

SEED = 42


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


# ============================================================
# DATASET
# ============================================================

class SequenceDataset(Dataset):

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        self.X = torch.tensor(
            X,
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# SEQUENCE BUILDING
# ============================================================

def build_sequences(
    df: pd.DataFrame,
):
    """
    Build one sequence per stock.

    Each sample contains the previous LOOKBACK observations
    ending at the prediction date.

    Shape:
        X = [samples, features, time]

    Target:
        target_up_5d
    """

    X_list = []
    y_list = []

    for symbol, g in df.groupby(
        "symbol",
        sort=False,
    ):

        g = g.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        values = (
            g[FEATURES]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        targets = (
            g["target_up_5d"]
            .to_numpy(dtype=np.float32)
        )

        if len(g) <= LOOKBACK:
            continue

        for end in range(
            LOOKBACK,
            len(g),
        ):

            start = end - LOOKBACK

            sequence = values[
                start:end
            ]

            target = targets[end]

            X_list.append(
                sequence.T
            )

            y_list.append(target)

    X = np.asarray(
        X_list,
        dtype=np.float32,
    )

    y = np.asarray(
        y_list,
        dtype=np.float32,
    )

    return X, y


# ============================================================
# TCN BUILDING BLOCK
# ============================================================

class Chomp1d(nn.Module):

    def __init__(self, chomp_size: int):
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
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()

        padding = (
            kernel_size - 1
        ) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
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
            out_channels,
            out_channels,
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
                in_channels,
                out_channels,
                kernel_size=1,
            )
            if in_channels != out_channels
            else None
        )

        self.final_relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):

        nn.init.kaiming_normal_(
            self.conv1.weight
        )

        nn.init.zeros_(
            self.conv1.bias
        )

        nn.init.kaiming_normal_(
            self.conv2.weight
        )

        nn.init.zeros_(
            self.conv2.bias
        )

        if self.downsample is not None:

            nn.init.kaiming_normal_(
                self.downsample.weight
            )

            nn.init.zeros_(
                self.downsample.bias
            )

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


# ============================================================
# TCN CLASSIFIER
# ============================================================

class TCNClassifier(nn.Module):

    def __init__(
        self,
        input_channels: int,
    ):
        super().__init__()

        channels = [
            64,
            64,
            128,
            128,
        ]

        kernel_size = 3

        dropout = 0.20

        layers = []

        in_channels = input_channels

        for i, out_channels in enumerate(
            channels
        ):

            dilation = 2 ** i

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout,
                )
            )

            in_channels = out_channels

        self.tcn = nn.Sequential(
            *layers
        )

        self.head = nn.Sequential(
            nn.Linear(
                channels[-1],
                64,
            ),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, x):

        # x:
        # [batch, features, time]

        out = self.tcn(x)

        # Last timestep representation.
        last = out[:, :, -1]

        logits = self.head(last)

        return logits.squeeze(1)


# ============================================================
# METRICS
# ============================================================

def evaluate_model(
    model,
    loader,
    device,
):

    model.eval()

    all_prob = []
    all_y = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True,
            )

            logits = model(
                X_batch
            )

            prob = torch.sigmoid(
                logits
            )

            all_prob.append(
                prob.detach()
                .cpu()
                .numpy()
            )

            all_y.append(
                y_batch.numpy()
            )

    prob = np.concatenate(
        all_prob
    )

    y = np.concatenate(
        all_y
    )

    pred = (
        prob >= 0.5
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            y,
            pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y,
            pred,
        ),
        "roc_auc": roc_auc_score(
            y,
            prob,
        ),
        "pr_auc": average_precision_score(
            y,
            prob,
        ),
        "brier": brier_score_loss(
            y,
            prob,
        ),
    }

    return metrics, prob, y


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0
    total_count = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            device,
            non_blocking=True,
        )

        y_batch = y_batch.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            X_batch
        )

        loss = criterion(
            logits,
            y_batch,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = len(
            y_batch
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_count += batch_size

    return (
        total_loss / total_count
    )


# ============================================================
# CROSS-SECTIONAL RANKING
# ============================================================

def evaluate_cross_sectional_ranking(
    df,
    probabilities,
    sequence_timestamps,
):

    x = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                sequence_timestamps,
                utc=True,
            ),
            "symbol": sequence_timestamps["symbol"]
            if isinstance(
                sequence_timestamps,
                pd.DataFrame,
            )
            else None,
            "probability": probabilities,
        }
    )

    # This function is intentionally kept simple.
    # Ranking is performed separately below using
    # the test metadata constructed in main().


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    print("=" * 70)
    print("TCN 5-DAY DIRECTION MODEL")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Rows   :",
        end=" ",
    )

    df = pd.read_parquet(
        INPUT
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values(
        [
            "symbol",
            "timestamp",
        ]
    ).reset_index(
        drop=True
    )

    print(len(df))

    print(
        "Symbols:",
        df["symbol"].nunique(),
    )

    print(
        "Features:",
        len(FEATURES),
    )

    print(
        "Lookback:",
        LOOKBACK,
    )

    # --------------------------------------------------------
    # BUILD SEQUENCES
    # --------------------------------------------------------

    print()
    print(
        "BUILDING SEQUENCES..."
    )

    X, y = build_sequences(
        df
    )

    print(
        "Sequences:",
        len(X),
    )

    # --------------------------------------------------------
    # BUILD EXACT METADATA
    # --------------------------------------------------------

    metadata = []

    for symbol, g in df.groupby(
        "symbol",
        sort=False,
    ):

        g = g.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        if len(g) <= LOOKBACK:
            continue

        for end in range(
            LOOKBACK,
            len(g),
        ):

            metadata.append(
                {
                    "symbol": symbol,
                    "timestamp": g.loc[
                        end,
                        "timestamp",
                    ],
                    "future_return_5d": g.loc[
                        end,
                        "future_return_5d",
                    ],
                }
            )

    metadata = pd.DataFrame(
        metadata
    )

    if len(metadata) != len(X):

        raise RuntimeError(
            "Sequence/metadata length mismatch: "
            f"{len(X)} vs {len(metadata)}"
        )

    # --------------------------------------------------------
    # TIME SPLIT
    # --------------------------------------------------------

    timestamps = pd.to_datetime(
        metadata["timestamp"],
        utc=True,
    )

    train_mask = (
        timestamps
        < pd.Timestamp(
            "2024-01-01",
            tz="UTC",
        )
    ).to_numpy()

    valid_mask = (
        (timestamps >= pd.Timestamp(
            "2024-01-01",
            tz="UTC",
        ))
        &
        (timestamps < pd.Timestamp(
            "2025-01-01",
            tz="UTC",
        ))
    ).to_numpy()

    test_mask = (
        timestamps
        >= pd.Timestamp(
            "2025-01-01",
            tz="UTC",
        )
    ).to_numpy()

    print()
    print("SPLIT")
    print("-" * 70)
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
    # CREATE DATASETS
    # --------------------------------------------------------

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print()
    print("TARGET")
    print("-" * 70)

    print(
        "Train target rate:",
        float(y_train.mean()),
    )

    print(
        "Valid target rate:",
        float(y_valid.mean()),
    )

    print(
        "Test target rate :",
        float(y_test.mean()),
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        SequenceDataset(
            X_train,
            y_train,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    valid_loader = DataLoader(
        SequenceDataset(
            X_valid,
            y_valid,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        SequenceDataset(
            X_test,
            y_test,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = TCNClassifier(
        len(FEATURES)
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print("MODEL")
    print("-" * 70)
    print(model)
    print(
        "Parameters:",
        f"{parameter_count:,}",
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    train_positive_rate = float(
        y_train.mean()
    )

    train_negative_rate = (
        1.0
        - train_positive_rate
    )

    pos_weight = (
        train_negative_rate
        / train_positive_rate
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_auc = -np.inf
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        valid_metrics, _, _ = evaluate_model(
            model,
            valid_loader,
            device,
        )

        scheduler.step(
            valid_metrics["roc_auc"]
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.6f} | "
            f"val_auc={valid_metrics['roc_auc']:.6f} | "
            f"val_pr={valid_metrics['pr_auc']:.6f} | "
            f"lr={current_lr:.2e}"
        )

        if (
            valid_metrics["roc_auc"]
            > best_auc
        ):

            best_auc = (
                valid_metrics["roc_auc"]
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
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

    test_metrics, test_prob, test_y = (
        evaluate_model(
            model,
            test_loader,
            device,
        )
    )

    print()
    print("-" * 70)
    print("TEST")
    print("-" * 70)

    print(
        "Accuracy          :",
        test_metrics["accuracy"],
    )

    print(
        "Balanced accuracy :",
        test_metrics[
            "balanced_accuracy"
        ],
    )

    print(
        "ROC-AUC           :",
        test_metrics["roc_auc"],
    )

    print(
        "PR-AUC            :",
        test_metrics["pr_auc"],
    )

    print(
        "Brier score       :",
        test_metrics["brier"],
    )

    # --------------------------------------------------------
    # CROSS-SECTIONAL RANKING
    # --------------------------------------------------------

    test_metadata = (
        metadata.loc[
            test_mask
        ].copy()
    )

    test_metadata[
        "probability"
    ] = test_prob

    test_metadata["rank"] = (
        test_metadata
        .groupby("timestamp")[
            "probability"
        ]
        .rank(
            ascending=False,
            method="first",
        )
    )

    print()
    print("=" * 70)
    print("CROSS-SECTIONAL RANKING")
    print("=" * 70)

    for k in [
        1,
        3,
        5,
        10,
    ]:

        top = test_metadata[
            test_metadata["rank"] <= k
        ]

        daily_return = (
            top.groupby("timestamp")[
                "future_return_5d"
            ]
            .mean()
        )

        mean_return = (
            daily_return.mean()
        )

        std_return = (
            daily_return.std()
        )

        if std_return > 0:

            sharpe = (
                np.sqrt(252 / 5)
                * mean_return
                / std_return
            )

        else:

            sharpe = np.nan

        hit_rate = (
            top["future_return_5d"]
            > 0
        ).mean()

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

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "features":
                FEATURES,
            "lookback":
                LOOKBACK,
            "model_type":
                "TCNClassifier",
            "test_metrics":
                test_metrics,
            "best_validation_auc":
                best_auc,
        },
        OUTPUT,
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        OUTPUT
    )


if __name__ == "__main__":
    main()
