from pathlib import Path

import duckdb


INPUT = Path("/workspace/data/raw/sample_nifty50.parquet")


def main() -> None:
    connection = duckdb.connect(":memory:")

    print("=== MODEL 1 MARKET DATA QUERY TEST ===")

    total_rows = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{INPUT}')
        """
    ).fetchone()[0]

    print("Total rows:", total_rows)

    summary = connection.execute(
        f"""
        SELECT
            symbol,
            MIN(timestamp) AS first_timestamp,
            MAX(timestamp) AS last_timestamp,
            MIN(low) AS minimum_price,
            MAX(high) AS maximum_price,
            AVG(volume) AS average_volume
        FROM read_parquet('{INPUT}')
        GROUP BY symbol
        """
    ).fetchdf()

    print()
    print(summary.to_string(index=False))

    connection.close()

    print()
    print("DUCKDB MARKET QUERY: SUCCESS")


if __name__ == "__main__":
    main()
