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
#Advance vr for fetching token from symbol input with even wrong symbol input
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
results = []


# Fixed duplicate stock name prompt issue
symbol_input = input("Enter stock symbol (e.g., RELIANCE, TCS): ").upper()
matches = difflib.get_close_matches(symbol_input, symbol_list, n=1, cutoff=0.6)

if matches:
    corrected_symbol = matches[0]
    print(f"✅ Found symbol: {corrected_symbol}")
    symbol_token = next((item['token'] for item in instruments if item['symbol'] == corrected_symbol and item['exch_seg'] == exchange), None)
    if not symbol_token:
        print(f"❌ Symbol '{corrected_symbol}' not found on {exchange}.")
        exit(1)
else:
    print("❌ No close match found. Exiting.")
    exit(1)

# Fetch and process data
df = fetch_historical_data(smartApi, symbol_token)
if df is not None:
    df = identify_pivot_points(df)
    print("\n✅ Data processed successfully. Found {} pivot points.".format(len(df[df['pivot'] != 0])))

backcandles = 20

for candleid in range(0, len(df)):
    maxim = np.array([])
    minim = np.array([])
    xxmin = np.array([])
    xxmax = np.array([])
    
    try:
        for i in range(candleid-backcandles, candleid+1):
            if df.iloc[i].pivot == 1:
                minim = np.append(minim, df.iloc[i].low)
                xxmin = np.append(xxmin, i)
            if df.iloc[i].pivot == 2:
                maxim = np.append(maxim, df.iloc[i].high)
                xxmax = np.append(xxmax, i)
        
        if (xxmax.size < 3 and xxmin.size < 3) or xxmax.size == 0 or xxmin.size == 0:
            continue
            
        # Check for constant values which would cause division by zero
        if len(set(minim)) == 1 or len(set(maxim)) == 1:
            continue
            
        # Check for NaN or infinite values
        if (np.isnan(minim).any() or np.isnan(maxim).any() or 
            np.isinf(minim).any() or np.isinf(maxim).any()):
            continue

        try:
            slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
            slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)
            
            # Check if we got valid results
            if (np.isnan(slmin) or np.isnan(slmax) or 
                np.isnan(rmin) or np.isnan(rmax)):
                continue
                
            if abs(rmax)>=0.9 and abs(rmin)>=0.9 and slmin>=0.0001 and slmax<=-0.0001:
                print(rmin, rmax, candleid)
                break
                
        except ValueError as ve:
            continue
            
    except Exception as e:
        print(f"Warning: Error in triangle detection for candle {candleid}: {str(e)}")
        continue
        
    if candleid % 1000 == 0:
        print(candleid)

# Added message to indicate whether triangle patterns were found or not
patterns = detect_triangle_pattern(df)
if patterns:
    print(f"🔺 Found {len(patterns)} triangle patterns!")
else:
    print("❌ No triangle patterns detected.")


#Us plotly to visualize the data , us plotly.graph_objects to create a candlestick chart, candleID and timestamp as x-axis, open, high, low, close as y-axis