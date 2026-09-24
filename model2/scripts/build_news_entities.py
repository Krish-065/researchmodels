from pathlib import Path
import pandas as pd

ROOT = Path.home() / "24DIT065" / "model2"

UNIVERSE = ROOT / "01_raw" / "market" / "experiment1_universe.csv"
OUT = ROOT / "01_raw" / "news" / "news_entities.csv"

u = pd.read_csv(UNIVERSE)

# IMPORTANT:
# Prefer unambiguous company names.
# Do NOT use short ambiguous tokens such as BEL, SBI, HUL, ITC, TCS, LT.
# Those can create large numbers of false positives in global news.

ALIASES = {
    "ADANIENT": [
        "Adani Enterprises",
    ],
    "ADANIPORTS": [
        "Adani Ports",
        "Adani Ports and Special Economic Zone",
    ],
    "APOLLOHOSP": [
        "Apollo Hospitals",
        "Apollo Hospitals Enterprise",
    ],
    "ASIANPAINT": [
        "Asian Paints",
    ],
    "AXISBANK": [
        "Axis Bank",
    ],
    "BAJAJ-AUTO": [
        "Bajaj Auto",
    ],
    "BAJAJFINSV": [
        "Bajaj Finserv",
    ],
    "BAJFINANCE": [
        "Bajaj Finance",
    ],
    "BEL": [
        "Bharat Electronics",
    ],
    "BHARTIARTL": [
        "Bharti Airtel",
        "Airtel India",
    ],
    "BPCL": [
        "Bharat Petroleum",
        "Bharat Petroleum Corporation",
    ],
    "BRITANNIA": [
        "Britannia Industries",
    ],
    "CIPLA": [
        "Cipla",
    ],
    "COALINDIA": [
        "Coal India",
        "Coal India Limited",
    ],
    "DIVISLAB": [
        "Divi's Laboratories",
        "Divis Laboratories",
    ],
    "DRREDDY": [
        "Dr Reddy's Laboratories",
        "Dr. Reddy's Laboratories",
    ],
    "EICHERMOT": [
        "Eicher Motors",
    ],
    "GRASIM": [
        "Grasim Industries",
    ],
    "HCLTECH": [
        "HCLTech",
        "HCL Technologies",
    ],
    "HDFCBANK": [
        "HDFC Bank",
    ],
    "HEROMOTOCO": [
        "Hero MotoCorp",
        "Hero Motocorp",
    ],
    "HINDALCO": [
        "Hindalco Industries",
        "Hindalco",
    ],
    "HINDUNILVR": [
        "Hindustan Unilever",
        "Hindustan Unilever Limited",
    ],
    "ICICIBANK": [
        "ICICI Bank",
    ],
    "INDIGO": [
        "InterGlobe Aviation",
        "IndiGo",
    ],
    "INDUSINDBK": [
        "IndusInd Bank",
    ],
    "INFY": [
        "Infosys",
        "Infosys Limited",
    ],
    "ITC": [
        "ITC Limited",
        "ITC Ltd",
    ],
    "JSWSTEEL": [
        "JSW Steel",
        "JSW Steel Limited",
    ],
    "KOTAKBANK": [
        "Kotak Mahindra Bank",
        "Kotak Mahindra",
    ],
    "LT": [
        "Larsen & Toubro",
        "Larsen and Toubro",
        "Larsen Toubro",
    ],
    "M&M": [
        "Mahindra & Mahindra",
        "Mahindra and Mahindra",
    ],
    "MARUTI": [
        "Maruti Suzuki",
        "Maruti Suzuki India",
    ],
    "NESTLEIND": [
        "Nestle India",
        "Nestlé India",
    ],
    "NTPC": [
        "NTPC Limited",
        "NTPC",
    ],
    "ONGC": [
        "Oil and Natural Gas Corporation",
        "ONGC",
    ],
    "POWERGRID": [
        "Power Grid Corporation of India",
        "Power Grid Corporation",
    ],
    "RELIANCE": [
        "Reliance Industries",
        "Reliance Industries Limited",
    ],
    "SBIN": [
        "State Bank of India",
    ],
    "SHRIRAMFIN": [
        "Shriram Finance",
    ],
    "SUNPHARMA": [
        "Sun Pharmaceutical Industries",
        "Sun Pharma",
    ],
    "TATACONSUM": [
        "Tata Consumer Products",
        "Tata Consumer",
    ],
    "TATASTEEL": [
        "Tata Steel",
        "Tata Steel Limited",
    ],
    "TCS": [
        "Tata Consultancy Services",
    ],
    "TECHM": [
        "Tech Mahindra",
        "Tech Mahindra Limited",
    ],
    "TITAN": [
        "Titan Company",
        "Titan Company Limited",
    ],
    "TRENT": [
        "Trent Limited",
        "Trent Ltd",
    ],
    "ULTRACEMCO": [
        "UltraTech Cement",
        "UltraTech Cement Limited",
    ],
    "WIPRO": [
        "Wipro",
        "Wipro Limited",
    ],
}

rows = []

for symbol in u["symbol"].astype(str).str.strip():

    aliases = ALIASES.get(symbol)

    if not aliases:
        raise ValueError(
            f"No alias definition for {symbol}"
        )

    rows.append({
        "symbol": symbol,
        "aliases": " | ".join(aliases),
    })

out = pd.DataFrame(rows)

assert len(out) == 49
assert out["symbol"].nunique() == 49

out.to_csv(
    OUT,
    index=False,
)

print("=" * 80)
print("SAFE NEWS ENTITY MAP")
print("=" * 80)
print("Stocks:", len(out))
print("Saved :", OUT)
print()
print(out.to_string(index=False))
print("=" * 80)
