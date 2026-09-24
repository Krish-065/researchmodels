from __future__ import annotations

import csv
import zipfile
from pathlib import Path


ROOT = Path("/workspace/data/raw/market/nse/bhavcopy")


def main() -> None:
    files = sorted(ROOT.glob("*.zip"))

    if not files:
        raise RuntimeError("No NSE bhavcopy ZIP files found.")

    print("NSE BHAVCOPY FILES:", len(files))

    for path in files:
        print()
        print("=" * 70)
        print("FILE:", path.name)
        print("SIZE:", path.stat().st_size, "bytes")

        with zipfile.ZipFile(path) as z:
            names = z.namelist()

            print("CONTENTS:", names)

            csv_names = [
                name for name in names
                if name.lower().endswith(".csv")
            ]

            if not csv_names:
                raise RuntimeError(f"No CSV found in {path}")

            csv_name = csv_names[0]

            with z.open(csv_name) as f:
                reader = csv.DictReader(
                    line.decode("utf-8-sig")
                    for line in f
                )

                rows = []

                for row in reader:
                    rows.append(row)

                    if len(rows) >= 5:
                        break

                print("COLUMNS:")
                print(reader.fieldnames)

                print()
                print("FIRST 5 ROWS:")

                for row in rows:
                    print(row)


if __name__ == "__main__":
    main()
