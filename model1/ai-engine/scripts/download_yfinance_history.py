from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import yfinance as yf


OUT = Path("/workspace/data/raw/market/yfinance")
OUT.mkdir(parents=True, exist_ok=True)

SYMBOLS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'LT.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'HCLTECH.NS', 'M&M.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'ADANIENT.NS', 'ADANIPORTS.NS', 'NTPC.NS', 'POWERGRID.NS', 'ONGC.NS', 'COALINDIA.NS', 'TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'TECHM.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'ASIANPAINT.NS', 'BAJAJFINSV.NS', 'INDUSINDBK.NS', 'GRASIM.NS', 'CIPLA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS', 'BRITANNIA.NS', 'TATAMOTORS.NS', 'TATACONSUM.NS', 'BPCL.NS', 'SHRIRAMFIN.NS', 'TRENT.NS', 'BEL.NS', 'INDIGO.NS']

START = "2010-01-01"
END = "2026-08-26"


def main():
    print("=" * 70)
    print("YFINANCE HISTORICAL NSE DOWNLOAD")
    print("=" * 70)

    for symbol in SYMBOLS:
        print(f"\nDownloading {symbol} ...")

        df = yf.download(
            symbol,
            start=START,
            end=END,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df.empty:
            print("WARNING: no data")
            continue

        # Flatten yfinance MultiIndex columns.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        rename = {
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adjusted_close",
        }

        df = df.rename(columns=rename)

        df["symbol"] = symbol.replace(".NS", "")

        wanted = [
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adjusted_close",
        ]

        df = df[[c for c in wanted if c in df.columns]]

        output = OUT / f"{symbol.replace('.NS', '')}_daily.parquet"
        df.to_parquet(output, index=False)

        print("Rows :", f"{len(df):,}")
        print("First:", df["timestamp"].min())
        print("Last :", df["timestamp"].max())
        print("File :", output)

        time.sleep(1)


if __name__ == "__main__":
    main()
