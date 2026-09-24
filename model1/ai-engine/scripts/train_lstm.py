from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


INPUT = Path("/workspace/data/features/nse_features.parquet")
MODEL_PATH = Path("/workspace/data/models/lstm_5d.pt")

SEED = 42
LOOKBACK = 60

BATCH_SIZE = 512
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.25

LR = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 30
PATIENCE = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        positions: list[tuple[int, int]],
    ):
        self.X = X
        self.y = y
        self.positions = positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):

        start, end = self.positions[idx]

        x = self.X[start:end]
        y = self.y[end - 1]

        return (
            torch.tensor(
                x,
                dtype=torch.float32,
            ),
            torch.tensor(
                y,
                dtype=torch.float32,
            ),
        )


class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last = output[:, -1, :]

        last = self.dropout(last)

        return self.head(last).squeeze(-1)


def build_sequences(df: pd.DataFrame):

    X_list = []
    y_list = []
    meta = []

    for symbol, g in df.groupby("symbol", sort=False):

        g = g.sort_values("timestamp").reset_index(drop=True)

        values = (
            g[FEATURES]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        targets = (
            g["target_up_5d"]
            .astype(np.float32)
            .to_numpy()
        )

        timestamps = g["timestamp"].to_numpy()

        offset = sum(len(x) for x in X_list)

        X_list.append(values)
        y_list.append(targets)

        for i in range(LOOKBACK - 1, len(g)):

            meta.append(
                (
                    symbol,
                    timestamps[i],
                    offset + i,
                )
            )

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    return X, y, meta


def make_positions(
    meta,
    timestamps,
):

    positions = []

    # The global concatenated array contains complete
    # per-symbol blocks. Recover sequences using metadata.
    current_symbol = None
    symbol_indices = []

    for symbol, timestamp, index in meta:

        if symbol != current_symbol:

            if symbol_indices:

                arr = symbol_indices

                for j in range(LOOKBACK - 1, len(arr)):
                    positions.append(
                        (
                            arr[j - LOOKBACK + 1],
                            arr[j] + 1,
                        )
                    )

            current_symbol = symbol
            symbol_indices = []

        symbol_indices.append(index)

    if symbol_indices:

        arr = symbol_indices

        for j in range(LOOKBACK - 1, len(arr)):
            positions.append(
                (
                    arr[j - LOOKBACK + 1],
                    arr[j] + 1,
                )
            )

    return positions


def build_dataset(df):

    blocks = []

    for symbol, g in df.groupby("symbol", sort=False):

        g = (
            g.sort_values("timestamp")
            .reset_index(drop=True)
        )

        X = (
            g[FEATURES]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        y = (
            g["target_up_5d"]
            .astype(np.float32)
            .to_numpy()
        )

        dates = g["timestamp"].to_numpy()

        # Sequence ending at row i.
        for i in range(LOOKBACK - 1, len(g)):

            blocks.append(
                {
                    "symbol": symbol,
                    "timestamp": dates[i],
                    "X": X[
                        i - LOOKBACK + 1:i + 1
                    ],
                    "y": y[i],
                }
            )

    return blocks


class BlockDataset(Dataset):

    def __init__(self, blocks):

        self.blocks = blocks

    def __len__(self):

        return len(self.blocks)

    def __getitem__(self, idx):

        b = self.blocks[idx]

        return (
            torch.from_numpy(b["X"]),
            torch.tensor(
                b["y"],
                dtype=torch.float32,
            ),
        )


def predict(model, loader):

    model.eval()

    probabilities = []
    targets = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)

            logits = model(X)

            prob = torch.sigmoid(logits)

            probabilities.extend(
                prob.detach()
                .cpu()
                .numpy()
            )

            targets.extend(
                y.numpy()
            )

    return (
        np.asarray(probabilities),
        np.asarray(targets),
    )


def evaluate(name, probabilities, targets):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    print(
        "Accuracy          :",
        accuracy_score(targets, predictions),
    )

    print(
        "Balanced accuracy :",
        balanced_accuracy_score(
            targets,
            predictions,
        ),
    )

    print(
        "ROC-AUC           :",
        roc_auc_score(
            targets,
            probabilities,
        ),
    )

    print(
        "PR-AUC            :",
        average_precision_score(
            targets,
            probabilities,
        ),
    )

    print(
        "Brier score       :",
        brier_score_loss(
            targets,
            probabilities,
        ),
    )


def ranking_evaluation(
    test_blocks,
    probabilities,
):

    rows = []

    for block, probability in zip(
        test_blocks,
        probabilities,
    ):

        rows.append(
            {
                "timestamp": block["timestamp"],
                "symbol": block["symbol"],
                "probability": probability,
                "target": block["y"],
            }
        )

    df = pd.DataFrame(rows)

    print()
    print("=" * 70)
    print("CROSS-SECTIONAL RANKING")
    print("=" * 70)

    for k in [1, 3, 5, 10]:

        df["rank"] = (
            df.groupby("timestamp")[
                "probability"
            ]
            .rank(
                ascending=False,
                method="first",
            )
        )

        top = df[df["rank"] <= k]

        hit_rate = top["target"].mean()

        print(
            f"TOP {k:2d} "
            f"hit rate: {hit_rate:.6f}"
        )


def main():

    seed_everything(SEED)

    print("=" * 70)
    print("LSTM 5-DAY DIRECTION MODEL")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    df = pd.read_parquet(INPUT)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)

    print()
    print("Rows   :", len(df))
    print("Symbols:", df["symbol"].nunique())
    print("Features:", len(FEATURES))
    print("Lookback:", LOOKBACK)

    # ------------------------------------------------------------
    # Build sequences
    # ------------------------------------------------------------

    print()
    print("BUILDING SEQUENCES...")

    blocks = build_dataset(df)

    print(
        "Sequences:",
        f"{len(blocks):,}",
    )

    # ------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------

    train_blocks = [
        b for b in blocks
        if b["timestamp"] < pd.Timestamp(
            "2024-01-01",
            tz="UTC",
        )
    ]

    valid_blocks = [
        b for b in blocks
        if (
            b["timestamp"] >= pd.Timestamp(
                "2024-01-01",
                tz="UTC",
            )
            and
            b["timestamp"] < pd.Timestamp(
                "2025-01-01",
                tz="UTC",
            )
        )
    ]

    test_blocks = [
        b for b in blocks
        if b["timestamp"] >= pd.Timestamp(
            "2025-01-01",
            tz="UTC",
        )
    ]

    print()
    print("SPLIT")
    print("-" * 70)

    print("Train:", f"{len(train_blocks):,}")
    print("Valid:", f"{len(valid_blocks):,}")
    print("Test :", f"{len(test_blocks):,}")

    # ------------------------------------------------------------
    # Normalize using TRAIN ONLY
    # ------------------------------------------------------------

    train_values = np.concatenate(
        [
            b["X"]
            for b in train_blocks
        ],
        axis=0,
    )

    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)

    std[std < 1e-8] = 1.0

    def normalize(blocks):

        normalized = []

        for b in blocks:

            x = (
                b["X"] - mean
            ) / std

            normalized.append(
                {
                    "symbol": b["symbol"],
                    "timestamp": b["timestamp"],
                    "X": x.astype(
                        np.float32
                    ),
                    "y": b["y"],
                }
            )

        return normalized

    train_blocks = normalize(train_blocks)
    valid_blocks = normalize(valid_blocks)
    test_blocks = normalize(test_blocks)

    # ------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------

    train_ds = BlockDataset(train_blocks)
    valid_ds = BlockDataset(valid_blocks)
    test_ds = BlockDataset(test_blocks)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = LSTMClassifier(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    print()
    print("MODEL")
    print("-" * 70)

    print(model)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameters:,}",
    )

    # ------------------------------------------------------------
    # Loss / optimizer
    # ------------------------------------------------------------

    positive = sum(
        b["y"]
        for b in train_blocks
    )

    negative = len(train_blocks) - positive

    pos_weight = (
        negative / positive
        if positive > 0
        else 1.0
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    best_auc = -np.inf
    best_state = None
    patience_counter = 0

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        losses = []

        for X, y in train_loader:

            X = X.to(
                DEVICE,
                non_blocking=True,
            )

            y = y.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(X)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            losses.append(
                loss.item()
            )

        train_loss = np.mean(losses)

        valid_prob, valid_y = predict(
            model,
            valid_loader,
        )

        valid_auc = roc_auc_score(
            valid_y,
            valid_prob,
        )

        valid_pr = average_precision_score(
            valid_y,
            valid_prob,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.6f} | "
            f"val_auc={valid_auc:.6f} | "
            f"val_pr={valid_pr:.6f}"
        )

        if valid_auc > best_auc:

            best_auc = valid_auc

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "Early stopping."
                )

                break

    # ------------------------------------------------------------
    # Restore best model
    # ------------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # ------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------

    test_prob, test_y = predict(
        model,
        test_loader,
    )

    evaluate(
        "TEST",
        test_prob,
        test_y,
    )

    ranking_evaluation(
        test_blocks,
        test_prob,
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "features": FEATURES,
            "lookback": LOOKBACK,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "mean": mean,
            "std": std,
            "best_validation_auc":
                best_auc,
        },
        MODEL_PATH,
    )

    print()
    print(
        "MODEL SAVED:",
        MODEL_PATH,
    )


if __name__ == "__main__":
    main()
