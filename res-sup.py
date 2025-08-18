from SmartApi import SmartConnect
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate
from watchlist import banking_and_finance, oil_and_gas
import time
import requests
import pyotp
import numpy as np

# ============ CONFIGURATION ============
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

# ------------------------
# Fetch candle data (ONE_DAY)
# ------------------------
def fetch_candle_data(symbol: str, days: int = 120) -> pd.DataFrame | None:
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

    retries = 3
    for attempt in range(retries):
        try:
            response = smartApi.getCandleData(params)
            time.sleep(TIME_DELAY)
            if 'data' not in response or not response['data']:
                print(f"No candle data for {symbol}")
                return None

            df = pd.DataFrame(response['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume']).reset_index(drop=True)
            df = _normalize_timestamp_utc_naive(df)
            return df
        except Exception as e:
            print(f"Error fetching candle data for {symbol} (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None

# ------------------------
# Swing detection (local highs/lows)
# ------------------------
def find_swing_points(df: pd.DataFrame, window: int = 2):
    """
    A bar is a swing high if its 'high' is the max in [i-window, i+window]
    A bar is a swing low  if its 'low'  is the min in [i-window, i+window]
    """
    highs_idx = []
    lows_idx = []
    n = len(df)
    for i in range(window, n - window):
        seg = df.iloc[i - window:i + window + 1]
        if df['high'].iloc[i] == seg['high'].max():
            highs_idx.append(i)
        if df['low'].iloc[i] == seg['low'].min():
            lows_idx.append(i)
    return highs_idx, lows_idx

# ------------------------
# Price clustering (merge close-by levels)
# ------------------------
def cluster_prices(prices: list[float], eps_pct: float = 0.25) -> list[float]:
    """
    Simple greedy clustering on price axis.
    eps_pct = max distance to cluster center in %.
    Returns representative price per cluster (median).
    """
    if not prices:
        return []
    prices = sorted(prices)
    clusters = []
    current = [prices[0]]

    def within(p, center):
        return abs(p - center) <= center * (eps_pct / 100.0)

    for p in prices[1:]:
        center = np.median(current)
        if within(p, center):
            current.append(p)
        else:
            clusters.append(np.median(current))
            current = [p]
    clusters.append(np.median(current))
    return clusters

# ------------------------
# Touch counter
# ------------------------
def count_touches(df: pd.DataFrame, level_price: float, proximity_pct: float = 0.5) -> int:
    """
    Counts bars where high/low/close is within proximity_pct of level_price.
    """
    tol = level_price * (proximity_pct / 100.0)
    hi_touch = (abs(df['high'] - level_price) <= tol)
    lo_touch = (abs(df['low'] - level_price) <= tol)
    cl_touch = (abs(df['close'] - level_price) <= tol)
    return int((hi_touch | lo_touch | cl_touch).sum())

# ------------------------
# Detect multiple supports/resistances with touch counts
# ------------------------
def detect_support_resistance_levels(
    df: pd.DataFrame,
    swing_window: int = 2,
    cluster_eps_pct: float = 0.25,
    proximity_pct: float = 0.5,
    min_touches: int = 2,
    include_extremes: bool = True,
):
    """
    Returns two lists of dicts (sorted by touches desc):
      supports: [{'price': p, 'touches': k}]
      resistances: [{'price': p, 'touches': k}]
    """
    if df is None or df.empty:
        return [], []

    swing_highs, swing_lows = find_swing_points(df, window=swing_window)

    # Candidate prices from swings
    high_prices = [float(df['high'].iloc[i]) for i in swing_highs]
    low_prices  = [float(df['low'].iloc[i]) for i in swing_lows]

    # Optionally include absolute extremes (ensures we consider the highest/lowest reached)
    if include_extremes:
        if len(df):
            high_prices.append(float(df['high'].max()))
            low_prices.append(float(df['low'].min()))

    # Cluster close-by levels to avoid duplicates
    res_levels = cluster_prices(high_prices, eps_pct=cluster_eps_pct)
    sup_levels = cluster_prices(low_prices, eps_pct=cluster_eps_pct)

    # Count touches
    resistances = []
    for p in res_levels:
        t = count_touches(df, p, proximity_pct=proximity_pct)
        if t >= min_touches:
            resistances.append({'price': round(float(p), 2), 'touches': int(t)})

    supports = []
    for p in sup_levels:
        t = count_touches(df, p, proximity_pct=proximity_pct)
        if t >= min_touches:
            supports.append({'price': round(float(p), 2), 'touches': int(t)})

    # Sort by most tested
    resistances.sort(key=lambda x: (-x['touches'], -x['price']))
    supports.sort(key=lambda x: (-x['touches'], x['price']))

    return supports, resistances

# ------------------------
# Pretty helpers for table
# ------------------------
def fmt_levels(levels: list[dict], top_n: int = 3) -> str:
    """
    "price:touches" CSV for top N levels (e.g., "412.5:5, 405.0:4, 398.0:3")
    """
    if not levels:
        return ""
    return ", ".join(f"{lvl['price']}:{lvl['touches']}" for lvl in levels[:top_n])

def nearest_level_info(current_price: float, supports: list[dict], resistances: list[dict]) -> tuple[str, float]:
    """
    Returns ("Support"/"Resistance"/"", distance_pct) for the nearest level to current_price.
    """
    candidates = []
    for lvl in supports:
        candidates.append(("Support", lvl['price'], abs(current_price - lvl['price']) / lvl['price'] * 100.0))
    for lvl in resistances:
        candidates.append(("Resistance", lvl['price'], abs(current_price - lvl['price']) / lvl['price'] * 100.0))
    if not candidates:
        return "", float('inf')
    t, p, d = min(candidates, key=lambda x: x[2])
    return f"{t} @ {p}", d

# ------------------------
# Analyzer (no S1; pure SR levels)
# ------------------------
def analyze_levels(
    df: pd.DataFrame,
    window_days: int = 84,
    swing_window: int = 2,
    cluster_eps_pct: float = 0.25,
    proximity_pct: float = 0.5,
    min_touches: int = 2,
    top_n: int = 3,
    near_flag_pct: float = 1.0,
):
    """
    Returns a dict for the table:
      {
        "Current Price": ...,
        "Top Resistances": "price:touches, ...",
        "Top Supports":    "price:touches, ...",
        "Nearest Level":   "Support @ X" or "Resistance @ Y",
        "Nearest Dist %":  distance in %
      }
    """
    if df is None or len(df) < 10:
        return None

    dfw = filter_by_calendar_window(df.copy(), window_days)
    if len(dfw) < 10:
        return None

    current_price = float(dfw['close'].iloc[-1])

    supports, resistances = detect_support_resistance_levels(
        dfw,
        swing_window=swing_window,
        cluster_eps_pct=cluster_eps_pct,
        proximity_pct=proximity_pct,
        min_touches=min_touches,
        include_extremes=True
    )

    if not supports and not resistances:
        return None

    nearest_desc, nearest_dist = nearest_level_info(current_price, supports, resistances)

    # Optional "near" flag if within near_flag_pct to ANY level
    def is_near_any(levels):
        for lvl in levels:
            if abs(current_price - lvl['price']) / lvl['price'] * 100.0 <= near_flag_pct:
                return True
        return False

    zone = []
    if is_near_any(resistances):
        zone.append("Near Resistance")
    if is_near_any(supports):
        zone.append("Near Support")
    zone_str = " & ".join(zone) if zone else ""

    return {
        "Current Price": round(current_price, 2),
        "Top Resistances": fmt_levels(resistances, top_n=top_n),
        "Top Supports": fmt_levels(supports, top_n=top_n),
        "Nearest Level": nearest_desc,
        "Nearest Dist %": round(nearest_dist, 2) if np.isfinite(nearest_dist) else None,
        "Zone": zone_str
    }

# ------------------------
# Main loop
# ------------------------
LOOKBACK_CALENDAR_DAYS_TO_FETCH = 180
ANALYZER_WINDOW_DAYS = 84
delay = 0.5

def main():
    result_table = []
    watchlist = banking_and_finance + oil_and_gas
    stock_count = 0

    for symbol in watchlist:
        time.sleep(delay)
        try:
            df = fetch_candle_data(symbol, days=LOOKBACK_CALENDAR_DAYS_TO_FETCH)
            result = analyze_levels(
                df,
                window_days=ANALYZER_WINDOW_DAYS,
                swing_window=2,           # pivot strength
                cluster_eps_pct=0.25,     # cluster merge threshold (% of price)
                proximity_pct=0.5,        # count touches within ±0.5%
                min_touches=2,            # require at least 2 tests
                top_n=3,
                near_flag_pct=1.0,        # "near" if within 1% of a level
            )
            if result:
                result_table.append([symbol] + list(result.values()))
                stock_count += 1
        except Exception as e:
            # Keep row width consistent with headers below (6 fields after symbol)
            result_table.append([symbol, f"Error: {str(e)}"] + [""] * 5)

    print(f"Total individual stocks processed: {stock_count}")
    headers = [
        "Symbol",
        "Current Price",
        "Top Resistances",
        "Top Supports",
        "Nearest Level",
        "Nearest Dist %",
        "Zone",
    ]
    print(tabulate(result_table, headers=headers, tablefmt="pretty"))

if __name__ == "__main__":
    main()
