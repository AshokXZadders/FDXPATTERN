import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tabulate import tabulate
from watchlist import (
    banking_and_finance,automobiles,oil_and_gas,
    it_and_services,pharmaceuticals,metals_and_mining,chemicals,
    construction_and_cement,consumer_goods,utilities,real_estate,
    telecom,media,retail,capital_goods_and_engineering,transportation_and_logistics,
    hospital_and_healthcare,miscellaneous
)
import pyotp
import requests
from SmartApi import SmartConnect
import time

API_KEY = ""
CLIENT_CODE = ""
PASSWORD = ""
TOTP_SECRET = ""
EXCHANGE = "NSE"
TIME_DELAY = 0.5  # seconds delay between API calls

# ============ LOGIN ============
smartApi = SmartConnect(api_key=API_KEY)
totp = pyotp.TOTP(TOTP_SECRET).now()
login_data = smartApi.generateSession(CLIENT_CODE, PASSWORD, totp)
AUTH_TOKEN = login_data['data']['jwtToken']
FEED_TOKEN = smartApi.getfeedToken()

# Load instrument master
print("Fetching instrument master list...")
response = requests.get(
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)
instruments = response.json()

symbol_to_token = {
    item['symbol']: item['token']
    for item in instruments if item['exch_seg'] == EXCHANGE
}

# ------------------------
# Utility: normalize timestamp
# ------------------------
def _normalize_timestamp_utc_naive(df: pd.DataFrame) -> pd.DataFrame:
    # Parse as tz-aware UTC, then drop tz to make comparisons safe
    ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    return df

def filter_by_calendar_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(df['timestamp']):
        df = _normalize_timestamp_utc_naive(df)
    cutoff = datetime.utcnow() - timedelta(days=days)
    return df[df['timestamp'] >= cutoff].reset_index(drop=True)

# ======== MODIFY fetch_candle_data for 5-min candles =========
def fetch_candle_data(symbol: str, days: int = 5) -> pd.DataFrame | None:
    token = symbol_to_token.get(symbol)
    if not token:
        print(f"Token not found for {symbol}")
        return None

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    params = {
        "exchange": EXCHANGE,
        "symboltoken": str(token),
        "interval": "FIVE_MINUTE",   # changed from ONE_DAY
        "fromdate": start.strftime('%Y-%m-%d %H:%M'),
        "todate": end.strftime('%Y-%m-%d %H:%M')
    }

    try:
        response = smartApi.getCandleData(params)
        time.sleep(TIME_DELAY)
        if 'data' not in response or not response['data']:
            print(f"No candle data for {symbol}")
            return None

        df = pd.DataFrame(response['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        df = _normalize_timestamp_utc_naive(df)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# ======== MACD(8,21,5) + EMA200 Strategy =========
def apply_macd_strategy(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    # Calculate EMAs
    df['ema8'] = df['close'].ewm(span=8, adjust=False).mean() #change from 12 to 8
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean() # change from 26 to 21
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

    # MACD line
    df['macd_line'] = df['ema8'] - df['ema21']
    # Signal line (5 EMA of MACD line)
    df['signal_line'] = df['macd_line'].ewm(span=5, adjust=False).mean()  # change from 9 to 5
    # Histogram
    df['histogram'] = df['macd_line'] - df['signal_line']

    # Generate signal
    signal = None
    if len(df) > 1:
        prev = df.iloc[-2]
        curr = df.iloc[-1]

        # Bullish crossover
        if prev['macd_line'] < prev['signal_line'] and curr['macd_line'] > curr['signal_line']:
            if curr['close'] > curr['ema200']:
                signal = "BUY"

        # Bearish crossover
        elif prev['macd_line'] > prev['signal_line'] and curr['macd_line'] < curr['signal_line']:
            if curr['close'] < curr['ema200']:
                signal = "SELL"

    return {
        "last_price": df['close'].iloc[-1],
        "macd": df['macd_line'].iloc[-1],
        "signal": df['signal_line'].iloc[-1],
        "histogram": df['histogram'].iloc[-1],
        "ema200": df['ema200'].iloc[-1],
        "action": signal
    }

# ======== Run Screener on Watchlist =========
def run_screener(watchlist, name="Watchlist"):
    results = []
    for symbol in watchlist:
        df = fetch_candle_data(symbol, days=5)
        res = apply_macd_strategy(df)
        if res:
            results.append([symbol, res['last_price'], res['action'], 
                            round(res['macd'], 2), round(res['signal'], 2), round(res['histogram'], 2)])
    
    if results:
        print(f"\n{name} Results:")
        print(tabulate(results, headers=["Symbol", "Last Price", "Signal", "MACD", "Signal Line", "Histogram"],tablefmt="pretty"))
    else:
        print(f"\n{name}: No signals found")

# ======== Execute =========    
if __name__ == "__main__":
    run_screener(banking_and_finance, "Banking & Finance")
    run_screener(oil_and_gas, "Oil & Gas")
    run_screener(automobiles, "Automobiles")
    run_screener(it_and_services, "IT & Services")  
    run_screener(pharmaceuticals, "Pharmaceuticals")
    run_screener(metals_and_mining, "Metals & Mining")      
    run_screener(chemicals, "Chemicals")
    run_screener(construction_and_cement, "Construction & Cement")
    run_screener(consumer_goods, "Consumer Goods")
    run_screener(utilities, "Utilities")
    run_screener(real_estate, "Real Estate")
    run_screener(telecom, "Telecom")
    run_screener(media, "Media")
    run_screener(retail, "Retail")
    run_screener(capital_goods_and_engineering, "Capital Goods & Engineering")
    run_screener(transportation_and_logistics, "Transportation & Logistics")
    run_screener(hospital_and_healthcare, "Hospital & Healthcare")
    run_screener(miscellaneous, "Miscellaneous")
    print("\nScreener execution completed.")

