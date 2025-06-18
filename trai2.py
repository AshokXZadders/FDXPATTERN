import requests
import pyotp
import json
import time
import datetime
from SmartApi import SmartConnect
import pandas as pd
import difflib
import numpy as np
from scipy.stats import linregress
# Importing watchlists from a separate file named watchlist.py
from watchlist import banking_and_finance, automobiles, oil_and_gas, it_and_services, pharmaceuticals, metals_and_mining, chemicals, construction_and_cement, consumer_goods, utilities, real_estate, telecom, media, retail, capital_goods_and_engineering, transportation_and_logistics, hospital_and_healthcare, miscellaneous

# === Angel Broking API Setup ===
API_KEY = ""
CLIENT_CODE = ""
PASSWORD = ""
TOTP_SECRET = ""

def initialize_api():
    try:
        smartApi = SmartConnect(api_key=API_KEY)
        totp = pyotp.TOTP(TOTP_SECRET).now()
        
        login_data = smartApi.generateSession(CLIENT_CODE, PASSWORD, totp)
        if not login_data or 'data' not in login_data:
            raise Exception("Failed to login: Invalid response from API")
            
        auth_token = login_data['data']['jwtToken']
        feed_token = smartApi.getfeedToken()
        
        return smartApi, auth_token, feed_token
    except Exception as e:
        print(f"❌ Error during API initialization: {str(e)}")
        exit(1)
#symbole token finder 
def get_symbol_token(symbol_input, exchange="NSE"):
    try:
        response = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
        response.raise_for_status()  # Raises an HTTPError for bad responses
        instruments = response.json()
        
        symbol_list = [inst['symbol'] for inst in instruments if inst['exch_seg'] == exchange]
        matches = difflib.get_close_matches(symbol_input.upper(), symbol_list, n=1, cutoff=0.6)
        
        if not matches:
            print("❌ No matching symbol found.")
            return None
            
        corrected_symbol = matches[0]
        print(f"✅ Found symbol: {corrected_symbol}")
        
        for item in instruments:
            if item['symbol'] == corrected_symbol and item['exch_seg'] == exchange:
                return item['token']
                
        print("❌ Symbol found but token not available")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error while fetching instruments: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing instrument data: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return None
# Fetch historical data for a given symbol token
def fetch_historical_data(smartApi, symbol_token, days=15, interval="ONE_HOUR"):
    try:
        print("📊 Fetching market data...")
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=days)

        params = {
            "exchange": "NSE",
            "symboltoken": str(symbol_token),
            "interval": interval,
            "fromdate": start.strftime('%Y-%m-%d %H:%M'),
            "todate": end.strftime('%Y-%m-%d %H:%M')
        }

        response = smartApi.getCandleData(params)

        if not isinstance(response, dict) or 'data' not in response or not response['data']:
            print("⚠️ No data available for this symbol")
            return None

        df = pd.DataFrame(response['data'], 
                         columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Clean and process the data
        df = df[df['volume'] != 0]  # Remove zero volume candles
        df.reset_index(drop=True, inplace=True)
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Convert price columns to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        if len(df) < 60:
            print("⚠️ Not enough data points for analysis")
            return None
            
        print("✅ Data loaded successfully")
        return df

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def identify_pivot_points(df, left_bars=3, right_bars=3):
    """
    Identifies pivot points (swing highs and lows) in the price data
    Returns: 0 for no pivot, 1 for pivot low, 2 for pivot high, 3 for both
    """
    try:
        def pivotid(df1, l, n1, n2):
            if l-n1 < 0 or l+n2 >= len(df1):
                return 0
            
            window_low = df1.iloc[l-n1:l+n2+1]['low'].values
            window_high = df1.iloc[l-n1:l+n2+1]['high'].values
            
            current_low = df1.iloc[l]['low']
            current_high = df1.iloc[l]['high']
            
            pivot_low = all(current_low <= window_low[i] for i in range(len(window_low)) if i != n1)
            pivot_high = all(current_high >= window_high[i] for i in range(len(window_high)) if i != n1)
            
            if pivot_low and pivot_high:
                return 3
            elif pivot_low:
                return 1
            elif pivot_high:
                return 2
            return 0
        
        df['pivot'] = 0
        for i in range(len(df)):
            df.iloc[i, df.columns.get_loc('pivot')] = pivotid(df, i, left_bars, right_bars)
        
        df['pointpos'] = np.nan
        mask_low = df['pivot'] == 1
        mask_high = df['pivot'] == 2
        
        df.loc[mask_low, 'pointpos'] = df.loc[mask_low, 'low'] - 1e-3
        df.loc[mask_high, 'pointpos'] = df.loc[mask_high, 'high'] + 1e-3
        
        print(f"📍 Found {len(df[df['pivot'] > 0])} pivot points")
        return df
        
    except Exception as e:
        print(f"❌ Error while finding pivot points: {str(e)}")
        return df
#trignle pattern detection code
def detect_triangle_pattern(df, backcandles=20, min_r_value=0.7):
    """
    Detects triangle patterns in the price data
    Returns: List of detected patterns with their properties
    """
    patterns = []
    
    for candleid in range(backcandles, len(df)):
        window = df.iloc[candleid-backcandles:candleid+1]
        
        # Get pivot highs and lows
        highs = window[window['pivot'] == 2]
        lows = window[window['pivot'] == 1]
        
        if len(highs) < 3 or len(lows) < 3:
            continue
            
        # Added checks to handle NaN or Inf values in data before linregress
        lows = lows.dropna()
        highs = highs.dropna()

        if lows.empty or highs.empty:
            continue

        # Check for constant values which would cause division by zero
        if len(set(lows['low'])) == 1 or len(set(highs['high'])) == 1:
            continue

        try:
            low_indices = np.arange(len(lows))
            high_indices = np.arange(len(highs))

            # Additional check for valid data
            if len(low_indices) != len(lows['low']) or len(high_indices) != len(highs['high']):
                continue

            # Calculate regression with error handling
            try:
                slmin, intercmin, rmin, pmin, semin = linregress(low_indices, lows['low'])
                slmax, intercmax, rmax, pmax, semax = linregress(high_indices, highs['high'])
            except ValueError:
                continue

            # Check if we got valid results
            if (np.isnan(slmin) or np.isnan(slmax) or 
                np.isnan(rmin) or np.isnan(rmax)):
                continue
            
            # Check for converging/diverging lines with good fit
            if (abs(rmax) >= min_r_value and abs(rmin) >= min_r_value):
                pattern_type = None
                
                # Symmetrical triangle
                if slmin > 0.0001 and slmax <= -0.0001:
                    pattern_type = "Symmetrical Triangle"
                # Ascending triangle
                elif abs(slmax) <= 0.0001 and slmin > 0.0001:
                    pattern_type = "Ascending Triangle"
                # Descending triangle
                elif abs(slmin) <= 0.0001 and slmax < -0.0001:
                    pattern_type = "Descending Triangle"
                    
                if pattern_type:
                    patterns.append({
                        'candleid': candleid,
                        'type': pattern_type,
                        'high_trend': {'slope': slmax, 'intercept': intercmax, 'r_value': rmax},
                        'low_trend': {'slope': slmin, 'intercept': intercmin, 'r_value': rmin}
                    })
    
        except Exception as e:
            print(f"Warning: Error in triangle detection for candle {candleid}: {str(e)}")
            continue
            
    return patterns

# Initialize API
smartApi, AUTH_TOKEN, FEED_TOKEN = initialize_api()



exchange = "NSE"
symbol_token = None



# === Get Instrument Token from Master File ===
response = requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json")
instruments = response.json()

symbol_list = [inst['symbol'] for inst in instruments if inst['exch_seg'] == 'NSE']

# Combine all watchlists into a single list
all_watchlists = banking_and_finance + automobiles + oil_and_gas + it_and_services + pharmaceuticals + metals_and_mining + chemicals + construction_and_cement + consumer_goods + utilities + real_estate + telecom + media + retail + capital_goods_and_engineering + transportation_and_logistics + hospital_and_healthcare + miscellaneous

results = []

# Remove single stock input and processing
for symbol_input in all_watchlists:
    matches = difflib.get_close_matches(symbol_input, symbol_list, n=1, cutoff=0.6)
    if matches:
        corrected_symbol = matches[0]
        print(f"\n=== Processing: {corrected_symbol} ===")
        symbol_token = next((item['token'] for item in instruments if item['symbol'] == corrected_symbol and item['exch_seg'] == exchange), None)
        if not symbol_token:
            print(f"❌ Symbol '{corrected_symbol}' not found on {exchange}.")
            continue
    else:
        print(f"❌ No close match found for {symbol_input}. Skipping.")
        continue

    df = fetch_historical_data(smartApi, symbol_token)
    time.sleep(1)  # To avoid hitting API rate limits
    if df is not None:
        df = identify_pivot_points(df)
        print("\n✅ Data processed successfully. Found {} pivot points.".format(len(df[df['pivot'] != 0])))
        patterns = detect_triangle_pattern(df)
        if patterns:
            print(f"🔺 Found {len(patterns)} triangle patterns for {corrected_symbol}!")
            results.append({'symbol': corrected_symbol, 'patterns': patterns, 'df': df})
        else:
            print(f"❌ No triangle patterns detected for {corrected_symbol}.")
    else:
        print(f"❌ No data for {corrected_symbol}.")

print(f"\nSummary: {sum(len(r['patterns']) for r in results)} triangle patterns detected across {len(results)} stocks.")
for res in results:
    symbol = res['symbol']
    df = res['df']
    for pattern in res['patterns']:
        candleid = pattern['candleid']
        try:
            pattern_time = df.iloc[candleid].name if candleid < len(df) else 'N/A'
            close_price = df.iloc[candleid]['close'] if candleid < len(df) else 'N/A'
        except Exception:
            pattern_time = 'N/A'
            close_price = 'N/A'
        high_r = pattern['high_trend']['r_value']
        low_r = pattern['low_trend']['r_value']
        high_slope = pattern['high_trend']['slope']
        low_slope = pattern['low_trend']['slope']
        print(f"{symbol} | {pattern['type']} | CandleID: {candleid} | Time: {pattern_time} | Close: {close_price:.2f} | HighR: {high_r:.3f} | LowR: {low_r:.3f} | HighSlope: {high_slope:.5f} | LowSlope: {low_slope:.5f}")


