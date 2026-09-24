from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# CONFIG
# =============================================================================

BASE = Path("/home/administrator/24DIT065/model2")
SEQ = BASE / "04_model/sequences"
MODEL_DIR = BASE / "04_model/models"
RESULT_DIR = BASE / "04_model/results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 60
N_FEATURES = 450

BATCH_SIZE = 512
EPOCHS = 50

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# GPU
# =============================================================================

print("=" * 110)
print("MODEL 2 — TCN + LSTM MULTI-TASK MODEL")
print("=" * 110)

print("\nTensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


# =============================================================================
# LOAD DATA
# =============================================================================

def load_split(prefix):

    X = np.load(
        SEQ / f"{prefix}_X.npy",
        mmap_mode="r"
    )

    y1 = np.load(
        SEQ / f"{prefix}_y_return_1d.npy",
        mmap_mode="r"
    )

    y3 = np.load(
        SEQ / f"{prefix}_y_return_3d.npy",
        mmap_mode="r"
    )

    y5 = np.load(
        SEQ / f"{prefix}_y_return_5d.npy",
        mmap_mode="r"
    )

    d1 = np.load(
        SEQ / f"{prefix}_y_direction_1d.npy",
        mmap_mode="r"
    )

    d5 = np.load(
        SEQ / f"{prefix}_y_direction_5d.npy",
        mmap_mode="r"
    )

    return X, y1, y3, y5, d1, d5


X_train, y1_train, y3_train, y5_train, d1_train, d5_train = (
    load_split("train_60d")
)

X_val, y1_val, y3_val, y5_val, d1_val, d5_val = (
    load_split("val_60d")
)

X_test, y1_test, y3_test, y5_test, d1_test, d5_test = (
    load_split("test_60d")
)


print("\nShapes:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)


# =============================================================================
# DATASETS
# =============================================================================
#
# tf.data avoids loading a second complete copy of the arrays into RAM.
# =============================================================================

def make_dataset(
    X,
    y1,
    y3,
    y5,
    d1,
    d5,
    shuffle=False,
):

    ds = tf.data.Dataset.from_tensor_slices(
        (
            X,
            {
                "return_1d": y1,
                "return_3d": y3,
                "return_5d": y5,
                "direction_1d": d1,
                "direction_5d": d5,
            },
        )
    )

    if shuffle:
        ds = ds.shuffle(
            min(len(X), 20000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds


train_ds = make_dataset(
    X_train,
    y1_train,
    y3_train,
    y5_train,
    d1_train,
    d5_train,
    shuffle=True,
)

val_ds = make_dataset(
    X_val,
    y1_val,
    y3_val,
    y5_val,
    d1_val,
    d5_val,
)

test_ds = make_dataset(
    X_test,
    y1_test,
    y3_test,
    y5_test,
    d1_test,
    d5_test,
)


# =============================================================================
# TCN BLOCK
# =============================================================================

def tcn_residual_block(
    x,
    filters,
    kernel_size,
    dilation_rate,
    dropout=0.15,
):

    residual = x

    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation=None,
    )(x)

    x = layers.LayerNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation=None,
    )(x)

    x = layers.LayerNormalization()(x)

    if residual.shape[-1] != filters:
        residual = layers.Conv1D(
            filters,
            kernel_size=1,
            padding="same",
        )(residual)

    x = layers.Add()(
        [x, residual]
    )

    x = layers.Activation(
        "gelu"
    )(x)

    return x


# =============================================================================
# MODEL
# =============================================================================

inputs = keras.Input(
    shape=(LOOKBACK, N_FEATURES),
    name="market_news_sequence",
)

x = layers.Conv1D(
    filters=128,
    kernel_size=3,
    padding="causal",
)(inputs)

x = layers.LayerNormalization()(x)
x = layers.Activation("gelu")(x)

for dilation in [
    1,
    2,
    4,
    8,
    16,
]:

    x = tcn_residual_block(
        x,
        filters=128,
        kernel_size=3,
        dilation_rate=dilation,
        dropout=0.15,
    )

# Temporal representation
x = layers.LSTM(
    128,
    return_sequences=False,
    dropout=0.15,
)(x)

x = layers.LayerNormalization()(x)

x = layers.Dense(
    128,
    activation="gelu",
)(x)

x = layers.Dropout(
    0.20
)(x)


# =============================================================================
# OUTPUT HEADS
# =============================================================================

return_1d = layers.Dense(
    1,
    name="return_1d",
)(x)

return_3d = layers.Dense(
    1,
    name="return_3d",
)(x)

return_5d = layers.Dense(
    1,
    name="return_5d",
)(x)

direction_1d = layers.Dense(
    1,
    activation="sigmoid",
    name="direction_1d",
)(x)

direction_5d = layers.Dense(
    1,
    activation="sigmoid",
    name="direction_5d",
)(x)


model = keras.Model(
    inputs=inputs,
    outputs=[
        return_1d,
        return_3d,
        return_5d,
        direction_1d,
        direction_5d,
    ],
)


# =============================================================================
# LOSS
# =============================================================================
#
# Huber is more robust than MSE for financial return tails.
# Binary cross-entropy handles the directional tasks.
# =============================================================================

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-4,
        clipnorm=1.0,
    ),
    loss={
        "return_1d":
            keras.losses.Huber(delta=0.01),
        "return_3d":
            keras.losses.Huber(delta=0.02),
        "return_5d":
            keras.losses.Huber(delta=0.03),
        "direction_1d":
            keras.losses.BinaryCrossentropy(),
        "direction_5d":
            keras.losses.BinaryCrossentropy(),
    },
    loss_weights={
        "return_1d": 1.0,
        "return_3d": 1.0,
        "return_5d": 1.0,
        "direction_1d": 0.50,
        "direction_5d": 0.50,
    },
    metrics={
        "return_1d": [
            keras.metrics.MeanAbsoluteError(
                name="mae"
            ),
        ],
        "return_3d": [
            keras.metrics.MeanAbsoluteError(
                name="mae"
            ),
        ],
        "return_5d": [
            keras.metrics.MeanAbsoluteError(
                name="mae"
            ),
        ],
        "direction_1d": [
            keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            keras.metrics.AUC(
                name="auc"
            ),
        ],
        "direction_5d": [
            keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            keras.metrics.AUC(
                name="auc"
            ),
        ],
    },
)


model.summary()


# =============================================================================
# CALLBACKS
# =============================================================================

best_model = MODEL_DIR / "tcn_lstm_50epoch_best.keras"

callbacks = [

    keras.callbacks.ModelCheckpoint(
        filepath=best_model,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),





    keras.callbacks.CSVLogger(
        RESULT_DIR / "tcn_lstm_50epoch_training_history.csv"
    ),
]


# =============================================================================
# TRAIN
# =============================================================================

print("\nStarting training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)


# =============================================================================
# SAVE FINAL MODEL
# =============================================================================

final_model = MODEL_DIR / "tcn_lstm_50epoch_final.keras"

model.save(
    final_model
)


# =============================================================================
# BEST MODEL
# =============================================================================

best = keras.models.load_model(
    best_model
)


# =============================================================================
# EVALUATION
# =============================================================================

print("\n" + "=" * 110)
print("TEST EVALUATION")
print("=" * 110)

test_results = best.evaluate(
    test_ds,
    return_dict=True,
    verbose=1,
)

for k, v in test_results.items():
    print(
        f"{k}: {float(v):.8f}"
    )


# =============================================================================
# PREDICTIONS
# =============================================================================

print("\nGenerating test predictions...")

pred = best.predict(
    test_ds,
    verbose=1,
)

(
    pred_y1,
    pred_y3,
    pred_y5,
    pred_d1,
    pred_d5,
) = pred

pred_y1 = pred_y1.reshape(-1)
pred_y3 = pred_y3.reshape(-1)
pred_y5 = pred_y5.reshape(-1)
pred_d1 = pred_d1.reshape(-1)
pred_d5 = pred_d5.reshape(-1)


# =============================================================================
# SAVE TEST PREDICTIONS
# =============================================================================

dates = np.load(
    SEQ / "test_60d_dates.npy"
)

stock_ids = np.load(
    SEQ / "test_60d_stock_id.npy"
)

mapping = pd.read_csv(
    SEQ / "stock_mapping.csv"
)

id_to_symbol = dict(
    zip(
        mapping["stock_id"],
        mapping["symbol"],
    )
)

symbols = [
    id_to_symbol[int(i)]
    for i in stock_ids
]

prediction_df = pd.DataFrame(
    {
        "Date": pd.to_datetime(dates),
        "symbol": symbols,

        "actual_return_1d":
            np.asarray(y1_test),

        "pred_return_1d":
            pred_y1,

        "actual_return_3d":
            np.asarray(y3_test),

        "pred_return_3d":
            pred_y3,

        "actual_return_5d":
            np.asarray(y5_test),

        "pred_return_5d":
            pred_y5,

        "actual_direction_1d":
            np.asarray(d1_test),

        "pred_direction_1d_prob":
            pred_d1,

        "actual_direction_5d":
            np.asarray(d5_test),

        "pred_direction_5d_prob":
            pred_d5,
    }
)

prediction_df = (
    prediction_df
    .sort_values(
        ["Date", "symbol"]
    )
    .reset_index(drop=True)
)

prediction_file = (
    RESULT_DIR /
    "tcn_lstm_50epoch_test_predictions.parquet"
)

prediction_df.to_parquet(
    prediction_file,
    index=False,
)


# =============================================================================
# SAVE TEST METRICS
# =============================================================================

metrics = {
    "test_loss": float(
        test_results["loss"]
    ),

    "test_return_1d_mae": float(
        test_results.get(
            "return_1d_mae",
            np.nan,
        )
    ),

    "test_return_3d_mae": float(
        test_results.get(
            "return_3d_mae",
            np.nan,
        )
    ),

    "test_return_5d_mae": float(
        test_results.get(
            "return_5d_mae",
            np.nan,
        )
    ),

    "test_direction_1d_accuracy": float(
        test_results.get(
            "direction_1d_accuracy",
            np.nan,
        )
    ),

    "test_direction_1d_auc": float(
        test_results.get(
            "direction_1d_auc",
            np.nan,
        )
    ),

    "test_direction_5d_accuracy": float(
        test_results.get(
            "direction_5d_accuracy",
            np.nan,
        )
    ),

    "test_direction_5d_auc": float(
        test_results.get(
            "direction_5d_auc",
            np.nan,
        )
    ),
}

pd.DataFrame(
    [metrics]
).to_csv(
    RESULT_DIR /
    "tcn_lstm_50epoch_test_metrics.csv",
    index=False,
)


# =============================================================================
# CONFIG
# =============================================================================

config = {
    "lookback": LOOKBACK,
    "features": N_FEATURES,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "seed": SEED,
    "architecture": "TCN + LSTM",
    "train_rows": len(X_train),
    "validation_rows": len(X_val),
    "test_rows": len(X_test),
}

with open(
    RESULT_DIR /
    "tcn_lstm_50epoch_config.json",
    "w",
) as f:
    json.dump(
        config,
        f,
        indent=2,
    )


print("\n" + "=" * 110)
print("STATUS: PASS — TCN + LSTM TRAINING COMPLETE")
print("=" * 110)

print("\nBest model:")
print(best_model)

print("\nFinal model:")
print(final_model)

print("\nPredictions:")
print(prediction_file)
