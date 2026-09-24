import yfinance as yf
from pathlib import Path

OUT = Path("01_raw/yahoo/historical")
OUT.mkdir(parents=True, exist_ok=True)

assets = [
    "^NSEI",
    "^INDIAVIX",
    "^GSPC",
    "^IXIC",
    "^DJI",
    "^N225",
    "^HSI",
    "GC=F",
    "CL=F",
    "USDINR=X",
    "BTC-USD",
    "ETH-USD",
]

for ticker in assets:

    print("=" * 60)
    print("Downloading:", ticker)

    try:

        df = yf.download(
            ticker,
            start="2019-01-01",
            interval="1d",
            auto_adjust=False,
            actions=True,
            repair=True,
            progress=False,
        )

        if len(df) == 0:
            print("NO DATA")
            continue

        df = df.reset_index()

        safe = (
            ticker
            .replace("^", "INDEX_")
            .replace("=", "_")
            .replace("-", "_")
        )

        path = OUT / f"{safe}_daily.parquet"

        df.to_parquet(path, index=False)

        print("Rows :", len(df))
        print("Saved:", path)

    except Exception as e:
        print("ERROR:", e)
