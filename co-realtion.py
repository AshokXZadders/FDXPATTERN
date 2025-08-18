from SmartApi import SmartConnect
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate
import time
import requests
import pyotp
import matplotlib.pyplot as plt

# ============ LOGIN ============
API_KEY = ""
CLIENT_CODE = ""
PASSWORD = ""
TOTP_SECRET = ""
EXCHANGE = "NSE"


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
    ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    return df

# ------------------------
# Fetch candle data (ONE_DAY)
# ------------------------
def fetch_candle_data(symbol: str, days: int = 8) -> pd.DataFrame | None:
    token = symbol_to_token.get(symbol)
    if not token:
        print(f"Token not found for {symbol}")
        return None

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    params = {
        "exchange": EXCHANGE,
        "symboltoken": str(token),
        "interval": "ONE_DAY",
        "fromdate": start.strftime('%Y-%m-%d %H:%M'),
        "todate": end.strftime('%Y-%m-%d %H:%M')
    }

    response = smartApi.getCandleData(params)
    

    if 'data' not in response or not response['data']:
        print(f"No candle data for {symbol}")
        return None

    df = pd.DataFrame(response['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume']).reset_index(drop=True)
    df = _normalize_timestamp_utc_naive(df)
    return df

# ------------------------
# Correlation & Signal Logic
# ------------------------
def calculate_correlation_signals(data: pd.DataFrame, asset1: str, asset2: str,
                                  window: int = 10, wide_window: int = 20, std_factor: float = 1.5) -> pd.DataFrame:
    df = data.copy()
    df['rolling_corr'] = df[asset1].rolling(window=window, min_periods=window).corr(df[asset2])
    df['avg_corr'] = df['rolling_corr'].rolling(window=wide_window, min_periods=5).mean()
    df['std_corr'] = df['rolling_corr'].rolling(window=wide_window, min_periods=5).std()

    df['upper_threshold'] = df['avg_corr'] + (std_factor * df['std_corr'])
    df['lower_threshold'] = df['avg_corr'] - (std_factor * df['std_corr'])

    df['signal'] = 0
    df.loc[df['rolling_corr'] > df['upper_threshold'], 'signal'] = -1
    df.loc[df['rolling_corr'] < df['lower_threshold'], 'signal'] = 2

    return df[['rolling_corr', 'avg_corr', 'upper_threshold', 'lower_threshold', 'signal']]

# ------------------------
# Combined Plot
# ------------------------
def plot_combined_data(price_data: pd.DataFrame, correlation_data: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot prices
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='black')
    ax1.plot(price_data.index, price_data.iloc[:, 0], label=price_data.columns[0], color='blue')
    ax1.plot(price_data.index, price_data.iloc[:, 1], label=price_data.columns[1], color='green')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot correlation on second axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Correlation', color='purple')
    ax2.plot(correlation_data.index, correlation_data['rolling_corr'], label='Rolling Corr', color='purple')
    ax2.plot(correlation_data.index, correlation_data['avg_corr'], '--', label='Avg Corr', color='orange')
    ax2.fill_between(correlation_data.index, correlation_data['lower_threshold'],
                     correlation_data['upper_threshold'], color='gray', alpha=0.2, label='Threshold')

    # Signals
    buy_signals = correlation_data[correlation_data['signal'] == 2]
    sell_signals = correlation_data[correlation_data['signal'] == -1]
    ax2.scatter(buy_signals.index, buy_signals['rolling_corr'], color='red', marker='^', label='Lower Signal', alpha=0.7)
    ax2.scatter(sell_signals.index, sell_signals['rolling_corr'], color='black', marker='v', label='Upper Signal', alpha=0.7)

    ax2.legend(loc='upper right')
    plt.title('Asset Prices & Rolling Correlation')
    plt.show()

# ------------------------
# Main Run
# ------------------------
s1_symbol = "HDFCBANK-EQ" # Example symbol use any valid symbol from the instrument master or Index token 
s2_symbol = "RELIANCE-EQ"

df1 = fetch_candle_data(s1_symbol, days=60)
df2 = fetch_candle_data(s2_symbol, days=60)

if df1 is not None and df2 is not None:
    # Merge close prices
    data = pd.concat([df1.set_index("timestamp")["close"], df2.set_index("timestamp")["close"]], axis=1, join="inner")
    data.columns = [s1_symbol, s2_symbol]

    # Calculate signals
    result = calculate_correlation_signals(data, s1_symbol, s2_symbol)

    # Plot
    plot_combined_data(data, result)
else:
    print("Could not fetch data for both symbols")
