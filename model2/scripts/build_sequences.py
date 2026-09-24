from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "24DIT065" / "model2"

DATA = ROOT / "04_model" / "datasets"
SPLITS = ROOT / "04_model" / "splits"
OUT = ROOT / "04_model" / "sequences"

OUT.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 60
TARGET = "target_return_1d"

FEATURES = pd.read_csv(
    DATA / "feature_list.csv"
)["feature"].tolist()

MEDIANS = pd.read_csv(
    DATA / "training_medians.csv"
)

# training_medians.csv has:
# Unnamed: 0 = feature name
# median      = training-set median

MEDIAN_MAP = dict(
    zip(
        MEDIANS["Unnamed: 0"],
        MEDIANS["median"]
    )
)

if set(FEATURES) != set(MEDIAN_MAP):
    missing = sorted(set(FEATURES) - set(MEDIAN_MAP))
    extra = sorted(set(MEDIAN_MAP) - set(FEATURES))

    raise ValueError(
        f"Feature/median mismatch. "
        f"Missing medians={missing[:10]}, "
        f"extra={extra[:10]}"
    )


def build_split(split_name):

    print()
    print("=" * 100)
    print(f"BUILDING {split_name.upper()} SEQUENCES")
    print("=" * 100)

    df = pd.read_parquet(
        SPLITS / f"{split_name}.parquet"
    )

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df.sort_values(["symbol", "Date"])
          .reset_index(drop=True)
    )

    required = FEATURES + [TARGET]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing[:20]}"
        )

    # --------------------------------------------------------
    # APPLY TRAINING-ONLY MEDIANS
    # --------------------------------------------------------

    print("Applying training-derived medians...")

    before_nan = df[FEATURES].isna().sum().sum()

    for feature in FEATURES:
        df[feature] = df[feature].fillna(
            MEDIAN_MAP[feature]
        )

    after_nan = df[FEATURES].isna().sum().sum()

    print("Feature NaN cells before:", before_nan)
    print("Feature NaN cells after :", after_nan)

    # Targets must never be imputed.
    if df[TARGET].isna().any():
        raise ValueError(
            f"{split_name}: target contains NaN"
        )

    # Replace/check infinities.
    feature_inf = np.isinf(
        df[FEATURES].to_numpy(
            dtype=np.float32
        )
    ).sum()

    target_inf = np.isinf(
        df[TARGET].to_numpy(
            dtype=np.float32
        )
    ).sum()

    print("Feature Inf cells:", feature_inf)
    print("Target Inf cells :", target_inf)

    if feature_inf > 0 or target_inf > 0:
        raise ValueError(
            f"{split_name}: infinite values detected"
        )

    if after_nan != 0:
        raise ValueError(
            f"{split_name}: NaNs remain after imputation"
        )

    X_list = []
    y_list = []

    stocks_used = []

    for symbol, g in df.groupby(
        "symbol",
        sort=True
    ):

        g = (
            g.sort_values("Date")
             .reset_index(drop=True)
        )

        if not g["Date"].is_monotonic_increasing:
            raise ValueError(
                f"Dates not monotonic: {symbol}"
            )

        if g["Date"].duplicated().any():
            raise ValueError(
                f"Duplicate dates: {symbol}"
            )

        X = g[FEATURES].to_numpy(
            dtype=np.float32
        )

        y = g[TARGET].to_numpy(
            dtype=np.float32
        )

        n = len(g)

        if n < SEQ_LEN:
            print(
                f"Skipping {symbol}: "
                f"only {n} rows"
            )
            continue

        stock_count = 0

        for end in range(
            SEQ_LEN - 1,
            n
        ):

            start = end - SEQ_LEN + 1

            window = X[
                start:end + 1
            ]

            target = y[end]

            if not np.isfinite(window).all():
                raise ValueError(
                    f"Non-finite window: "
                    f"{symbol}, "
                    f"{g.loc[end, 'Date']}"
                )

            if not np.isfinite(target):
                raise ValueError(
                    f"Non-finite target: "
                    f"{symbol}, "
                    f"{g.loc[end, 'Date']}"
                )

            X_list.append(window)
            y_list.append(target)

            stock_count += 1

        stocks_used.append(symbol)

        print(
            f"{symbol:15s} "
            f"rows={n:5d} "
            f"sequences={stock_count:5d}"
        )

    if len(X_list) == 0:
        raise ValueError(
            f"No sequences created for {split_name}"
        )

    X_out = np.stack(
        X_list
    ).astype(np.float32)

    y_out = np.asarray(
        y_list,
        dtype=np.float32
    )

    output_x = OUT / f"X_{split_name}_60d.npy"
    output_y = OUT / f"y_{split_name}_60d.npy"

    np.save(output_x, X_out)
    np.save(output_y, y_out)

    print()
    print("-" * 100)
    print(f"{split_name.upper()} RESULT")
    print("-" * 100)

    print("X shape:", X_out.shape)
    print("y shape:", y_out.shape)
    print("Stocks:", len(stocks_used))
    print("Features:", len(FEATURES))
    print("Sequence length:", SEQ_LEN)

    print("X dtype:", X_out.dtype)
    print("y dtype:", y_out.dtype)

    print(
        "X finite:",
        np.isfinite(X_out).all()
    )

    print(
        "y finite:",
        np.isfinite(y_out).all()
    )

    print("Saved:", output_x)
    print("Saved:", output_y)

    assert X_out.ndim == 3
    assert X_out.shape[1] == SEQ_LEN
    assert X_out.shape[2] == len(FEATURES)

    assert y_out.ndim == 1
    assert len(X_out) == len(y_out)

    assert len(stocks_used) == 49

    print()
    print("STATUS: PASS")


for split in [
    "train",
    "validation",
    "test",
]:
    build_split(split)

print()
print("=" * 100)
print("ALL SEQUENCE DATASETS CREATED")
print("=" * 100)
