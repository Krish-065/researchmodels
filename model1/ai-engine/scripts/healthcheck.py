from model1.paths import ensure_directories
from model1.reproducibility import seed_everything

import torch
import duckdb
import mlflow


def main() -> None:
    print("=== MODEL 1 HEALTH CHECK ===")

    ensure_directories()
    seed_everything(42)

    print("PyTorch :", torch.__version__)
    print("CUDA    :", torch.version.cuda)
    print("GPU     :", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Device  :", torch.cuda.get_device_name(0))

    connection = duckdb.connect(":memory:")
    result = connection.execute("SELECT 1 + 1").fetchone()
    connection.close()

    print("DuckDB  : SELECT 1 + 1 =", result[0])
    print("MLflow  :", mlflow.__version__)

    print()
    print("MODEL 1 HEALTH CHECK: SUCCESS")


if __name__ == "__main__":
    main()
