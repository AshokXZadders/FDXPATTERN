from datetime import datetime, timedelta
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import requests
import pyotp
from SmartApi import SmartConnect
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# CONFIGURATION
# =============================================================================

# Angel Broking API Credentials
API_KEY = ""
CLIENT_CODE = ""
PASSWORD = ""
TOTP_SECRET = ""
EXCHANGE = "NSE"

CONFIG = {
    "LOOKBACK_DAYS": 365,
    "INTERVAL": "ONE_HOUR",  # Options: ONE_DAY, ONE_WEEK, etc.
    
    # ATR-based Stop Loss Settings
    "ATR_LENGTH": 14,  # ATR calculation period
    "ATR_MULTIPLIER": 1.5,  # Multiplier for ATR stop loss
    
    # Risk Management
    "RISK_PERCENT": 1.0,  # Risk per trade as % of total capital (e.g., 1.0 = 1%)
    "RISK_REWARD_RATIO": 1.5,  # Target risk/reward ratio (e.g., 2.0 = 2:1)
    
    "USE_STOP_LOSS": True,  # Set to False to disable stop loss
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position: str  # 'long' or 'short'
    pnl: float
    pnl_percent: float
    points_captured: float  # Absolute points gained/lost
    exit_reason: str  # 'holding_period', 'stop_loss', 'risk_exceeded'
    stop_loss_price: float  # The stop loss price set for this trade
    atr_value: float  # ATR value at entry
    risk_amount: float  # Risk amount in rupees for this trade
    position_quantity: int  # Number of shares/quantity traded

@dataclass
class BacktestResult:
    """Contains backtesting results and metrics"""
    trades: List[Trade]
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_points_captured: float
    avg_points_per_trade: float
    avg_win: float
    avg_loss: float
    avg_pnl: float
    max_win: float
    max_loss: float
    profit_factor: float
    stop_loss_hits: int
    avg_risk_per_trade: float
    total_risk_taken: float
    risk_reward_ratio: float
    symbol_data: Dict[str, pd.DataFrame]  # Store price data for plotting
    
    def print_summary(self, show_yearly=True, show_per_stock=True, show_risk_analysis=True):
        """Print formatted summary of backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Losing Trades: {self.losing_trades}")
        print(f"Win Rate: {self.win_rate:.2f}%")
        print(f"\nP&L Metrics:")
        print(f"  Total P&L: ₹{self.total_pnl:.2f}")
        print(f"  Total Points Captured: {self.total_points_captured:.2f}")
        print(f"  Average Points per Trade: {self.avg_points_per_trade:.2f}")
        print(f"  Average P&L per Trade: ₹{self.avg_pnl:.2f}")
        print(f"  Average Win: ₹{self.avg_win:.2f}")
        print(f"  Average Loss: ₹{self.avg_loss:.2f}")
        print(f"  Max Win: ₹{self.max_win:.2f}")
        print(f"  Max Loss: ₹{self.max_loss:.2f}")
        print(f"  Profit Factor: {self.profit_factor:.2f}")
        print("="*60 + "\n")
        
        if show_risk_analysis and self.total_trades > 0:
            self._print_risk_analysis()
        
        if show_yearly and self.trades:
            self._print_yearly_breakdown()
        
        if show_per_stock and self.trades:
            self._print_stock_performance()
    
    def _print_risk_analysis(self):
        """Print risk management analysis"""
        print("RISK MANAGEMENT ANALYSIS")
        print("=" * 60)
        print(f"Stop Loss Hits: {self.stop_loss_hits} ({(self.stop_loss_hits/self.total_trades*100):.1f}% of trades)")
        print(f"Average Risk per Trade: ₹{self.avg_risk_per_trade:.2f}")
        print(f"Total Risk Taken: ₹{self.total_risk_taken:.2f}")
        print(f"Risk/Reward Ratio: 1:{self.risk_reward_ratio:.2f}")
        
        if CONFIG["USE_STOP_LOSS"]:
            print(f"\nStop Loss Settings:")
            print(f"  ATR Length: {CONFIG['ATR_LENGTH']} periods")
            print(f"  ATR Multiplier: {CONFIG['ATR_MULTIPLIER']}x")
            print(f"  Risk per Trade: {CONFIG['RISK_PERCENT']:.1f}% of capital")
            print(f"  Target Risk/Reward: 1:{CONFIG['RISK_REWARD_RATIO']:.1f}")
        else:
            print(f"Stop Loss: Disabled")
        print("=" * 60 + "\n")
    
    def _print_yearly_breakdown(self):
        """Print year-wise performance breakdown"""
        df = self.get_trades_df()
        if df.empty:
            return
        
        df['year'] = df['entry_date'].dt.year
        yearly_stats = []
        
        for year in sorted(df['year'].unique()):
            year_trades = df[df['year'] == year]
            total_trades = len(year_trades)
            wins = len(year_trades[year_trades['pnl'] > 0])
            losses = len(year_trades[year_trades['pnl'] <= 0])
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            total_pnl = year_trades['pnl'].sum()
            avg_return_pct = year_trades['pnl_percent'].mean()
            total_points = year_trades['points_captured'].sum()
            
            yearly_stats.append({
                'Year': year,
                'Trades': total_trades,
                'Wins': wins,
                'Losses': losses,
                'Win%': win_rate,
                'Total P&L': total_pnl,
                'Points': total_points,
                'Avg Return%': avg_return_pct
            })
        
        yearly_df = pd.DataFrame(yearly_stats)
        print("YEAR-WISE PERFORMANCE")
        print("=" * 95)
        print(yearly_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        print("=" * 95 + "\n")
    
    def _print_stock_performance(self):
        """Print per-stock performance in percentage terms"""
        df = self.get_trades_df()
        if df.empty:
            return
        
        stock_stats = []
        
        for symbol in sorted(df['symbol'].unique()):
            stock_trades = df[df['symbol'] == symbol]
            total_trades = len(stock_trades)
            wins = len(stock_trades[stock_trades['pnl'] > 0])
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            avg_return_pct = stock_trades['pnl_percent'].mean()
            total_return_pct = stock_trades['pnl_percent'].sum()
            total_points = stock_trades['points_captured'].sum()
            avg_points = stock_trades['points_captured'].mean()
            max_return_pct = stock_trades['pnl_percent'].max()
            min_return_pct = stock_trades['pnl_percent'].min()
            
            stock_stats.append({
                'Symbol': symbol.replace('-EQ', ''),
                'Trades': total_trades,
                'Win%': win_rate,
                'Points': total_points,
                'Avg Pts': avg_points,
                'Avg Return%': avg_return_pct,
                'Total Return%': total_return_pct,
                'Best%': max_return_pct,
                'Worst%': min_return_pct
            })
        
        # Sort by average return percentage
        stock_df = pd.DataFrame(stock_stats).sort_values('Avg Return%', ascending=False)
        
        print("PER-STOCK PERFORMANCE (% Returns & Points)")
        print("=" * 105)
        print(stock_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        print("=" * 105 + "\n")
    
    def get_trades_df(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis"""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = [{
            'symbol': t.symbol,
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'stop_loss_price': t.stop_loss_price,
            'atr_value': t.atr_value,
            'position': t.position,
            'quantity': t.position_quantity,
            'risk_amount': t.risk_amount,
            'pnl': t.pnl,
            'pnl_percent': t.pnl_percent,
            'points_captured': t.points_captured,
            'exit_reason': t.exit_reason
        } for t in self.trades]
        
        return pd.DataFrame(trades_data)
    
    def plot_trades(self, symbol: str = None, max_plots: int = 5):
        """
        Plot price charts with entry/exit points for trades
        
        Args:
            symbol: Specific symbol to plot (None = plot all up to max_plots)
            max_plots: Maximum number of symbols to plot
        """
        if not self.trades:
            print("No trades to plot!")
            return
        
        # Get unique symbols that have trades
        if symbol:
            symbols_to_plot = [symbol] if symbol in [t.symbol for t in self.trades] else []
        else:
            symbols_to_plot = list(set([t.symbol for t in self.trades]))[:max_plots]
        
        if not symbols_to_plot:
            print(f"No trades found for symbol: {symbol}")
            return
        
        for sym in symbols_to_plot:
            if sym not in self.symbol_data:
                print(f"No price data available for {sym}")
                continue
            
            symbol_trades = [t for t in self.trades if t.symbol == sym]
            if not symbol_trades:
                continue
            
            df = self.symbol_data[sym]
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Plot price
            ax.plot(df.index, df['Close'], label='Close Price', linewidth=1.5, color='black', alpha=0.7)
            
            # Plot each trade
            for trade in symbol_trades:
                color = 'green' if trade.pnl > 0 else 'red'
                marker_size = 100
                
                # Entry point
                ax.scatter(trade.entry_date, trade.entry_price, 
                          color=color, marker='^' if trade.position == 'long' else 'v',
                          s=marker_size, zorder=5, alpha=0.8,
                          label=f"Entry ({trade.position})" if trade == symbol_trades[0] else "")
                
                # Exit point
                ax.scatter(trade.exit_date, trade.exit_price,
                          color=color, marker='x', s=marker_size, zorder=5, alpha=0.8,
                          label="Exit" if trade == symbol_trades[0] else "")
                
                # Draw line connecting entry to exit
                ax.plot([trade.entry_date, trade.exit_date],
                       [trade.entry_price, trade.exit_price],
                       color=color, linestyle='--', linewidth=1, alpha=0.5)
                
                # Stop loss line
                if CONFIG["USE_STOP_LOSS"]:
                    ax.axhline(y=trade.stop_loss_price, color='orange', 
                             linestyle=':', linewidth=1, alpha=0.3)
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price (₹)', fontsize=12)
            ax.set_title(f'{sym.replace("-EQ", "")} - Trade Entry/Exit Points\n'
                        f'Total Trades: {len(symbol_trades)} | '
                        f'Win Rate: {len([t for t in symbol_trades if t.pnl > 0])/len(symbol_trades)*100:.1f}%',
                        fontsize=14, fontweight='bold')
            
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            print(f"\n📊 Chart displayed for {sym.replace('-EQ', '')}")

# =============================================================================
# ANGEL BROKING API CONNECTION
# =============================================================================

class AngelBrokingAPI:
    """Handles Angel Broking API connection and data fetching"""
    
    def __init__(self):
        self.smartApi = None
        self.symbol_to_token = {}
        self.login()
        self.load_instruments()
    
    def login(self):
        """Login to Angel Broking API"""
        try:
            self.smartApi = SmartConnect(api_key=API_KEY)
            totp = pyotp.TOTP(TOTP_SECRET).now()
            login_data = self.smartApi.generateSession(CLIENT_CODE, PASSWORD, totp)
            auth_token = login_data['data']['jwtToken']
            print("✓ Login Successful. Session ID generated.")
        except Exception as e:
            print(f"✗ Login Failed: {e}")
            raise
    
    def load_instruments(self):
        """Load instrument master list"""
        print("Fetching instrument master list...")
        try:
            response = requests.get(
                "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            )
            instruments = response.json()
            self.symbol_to_token = {
                item['symbol']: item['token'] 
                for item in instruments 
                if item['exch_seg'] == EXCHANGE
            }
            print(f"✓ Loaded {len(self.symbol_to_token)} instruments from {EXCHANGE}")
        except Exception as e:
            print(f"✗ Error fetching instrument master: {e}")
            raise
    
    def fetch_candle_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical candle data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE-EQ')
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        token = self.symbol_to_token.get(symbol)
        if not token:
            print(f"✗ Symbol {symbol} not found in instrument master")
            return None
        
        params = {
            "exchange": EXCHANGE,
            "symboltoken": str(token),
            "interval": CONFIG["INTERVAL"],
            "fromdate": start_date.strftime('%Y-%m-%d %H:%M'),
            "todate": end_date.strftime('%Y-%m-%d %H:%M')
        }
        
        try:
            response = self.smartApi.getCandleData(params)
            if not response or 'data' not in response:
                return None
            
            df = pd.DataFrame(
                response['data'], 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert to proper types
            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].apply(pd.to_numeric)
            
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Rename columns to match standard format
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"✗ Error fetching data for {symbol}: {e}")
            return None

# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """Dynamic backtesting framework for Angel Broking"""
    
    def __init__(self, api: AngelBrokingAPI):
        """
        Initialize backtester with Angel Broking API instance
        
        Args:
            api: AngelBrokingAPI instance for data fetching
        """
        self.api = api
        self.symbol_data_cache = {}  # Cache for price data
        
    def run_backtest(
        self,
        strategy_func: Callable,
        symbols: List[str],
        lookback_days: int = 365,
        holding_period_days: int = 2,
        position_type: str = 'long',
        initial_capital: float = 100000,
        position_size: float = 0.1
    ) -> BacktestResult:
        """
        Run backtest for a given strategy
        
        Args:
            strategy_func: Function that takes (DataFrame, symbol) and returns bool
            symbols: List of stock symbols to backtest (e.g., ['RELIANCE-EQ', 'TCS-EQ'])
            lookback_days: Number of days to backtest (default: 365)
            holding_period_days: Days to hold position (default: 5)
            position_type: 'long' or 'short' (default: 'long')
            initial_capital: Starting capital in INR (default: 100000)
            position_size: Fraction of capital per trade (default: 0.1)
            
        Returns:
            BacktestResult object with all trade details and metrics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer
        
        all_trades = []
        
        print(f"\n{'='*60}")
        print(f"Starting backtest for {len(symbols)} symbols")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"{'='*60}\n")
        
        # Iterate through each symbol
        for idx, symbol in enumerate(symbols, 1):
            print(f"[{idx}/{len(symbols)}] Processing {symbol}...", end=" ")
            
            try:
                # Fetch data
                df = self.api.fetch_candle_data(symbol, start_date, end_date)
                
                if df is None or df.empty:
                    print("✗ No data")
                    time.sleep(1.2)  # Rate limiting even on failure
                    continue
                
                # Cache the data for plotting
                self.symbol_data_cache[symbol] = df
                
                # Run strategy on this symbol
                trades = self._backtest_symbol(
                    df, 
                    symbol,
                    strategy_func,
                    holding_period_days,
                    position_type,
                    initial_capital,
                    position_size
                )
                
                all_trades.extend(trades)
                print(f"✓ {len(trades)} trades")
                
                # Rate limiting: 1.2 second delay between API calls
                time.sleep(1.2)
                
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                time.sleep(1.2)  # Rate limiting even on error
                continue
        
        # Calculate metrics
        result = self._calculate_metrics(all_trades)
        result.symbol_data = self.symbol_data_cache  # Attach data for plotting
        return result
    
    def _backtest_symbol(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy_func: Callable,
        holding_period: int,
        position_type: str,
        capital: float,
        position_size: float
    ) -> List[Trade]:
        """Backtest a single symbol with ATR-based stop loss and capital risk management"""
        trades = []
        i = 0
        
        # Calculate ATR for the entire dataset
        df['TR'] = pd.DataFrame({
            'hl': df['High'] - df['Low'],
            'hc': abs(df['High'] - df['Close'].shift(1)),
            'lc': abs(df['Low'] - df['Close'].shift(1))
        }).max(axis=1)
        df['ATR'] = df['TR'].rolling(window=CONFIG['ATR_LENGTH']).mean()
        
        while i < len(df) - holding_period:
            # Get data up to current point
            current_data = df.iloc[:i+1].copy()
            
            # Need enough data for ATR calculation
            if len(current_data) < CONFIG['ATR_LENGTH'] + 1:
                i += 1
                continue
            
            # Check if strategy gives signal
            try:
                strategy_result = strategy_func(current_data, symbol)
                
                # Handle both old (bool) and new (dict) return formats
                if isinstance(strategy_result, dict):
                    signal = strategy_result.get('signal', False)
                    strategy_position = strategy_result.get('position', 'long')
                else:
                    signal = strategy_result
                    strategy_position = position_type
                    
            except Exception as e:
                signal = False
                strategy_position = position_type
            
            if len(current_data) > 0 and signal:
                entry_date = df.index[i]
                entry_price = df.iloc[i]['Close']
                atr_value = df.iloc[i]['ATR']
                
                # Determine position type
                if position_type == 'both':
                    trade_position = strategy_position
                else:
                    trade_position = position_type
                
                # Calculate ATR-based stop loss
                if CONFIG["USE_STOP_LOSS"]:
                    atr_distance = atr_value * CONFIG['ATR_MULTIPLIER']
                    
                    if trade_position == 'long':
                        stop_loss_price = entry_price - atr_distance
                    else:  # short
                        stop_loss_price = entry_price + atr_distance
                    
                    # Calculate risk amount based on stop loss distance
                    risk_per_share = abs(entry_price - stop_loss_price)
                    
                    # Calculate maximum position size based on 1% capital risk
                    max_risk_amount = capital * (CONFIG['RISK_PERCENT'] / 100)
                    position_quantity = int(max_risk_amount / risk_per_share)
                    
                    # Skip trade if risk is too high (would require less than 1 share)
                    if position_quantity < 1:
                        i += 1
                        continue
                    
                    # Actual risk for this trade
                    actual_risk_amount = position_quantity * risk_per_share
                    
                else:
                    stop_loss_price = 0
                    position_quantity = int((capital * position_size) / entry_price)
                    actual_risk_amount = 0
                
                # Check each day until holding period or stop loss
                exit_idx = i + 1
                exit_reason = 'holding_period'
                hit_stop_loss = False
                
                while exit_idx <= min(i + holding_period, len(df) - 1):
                    current_price = df.iloc[exit_idx]['Low'] if trade_position == 'long' else df.iloc[exit_idx]['High']
                    
                    # Check stop loss
                    if CONFIG["USE_STOP_LOSS"]:
                        if trade_position == 'long' and current_price <= stop_loss_price:
                            exit_reason = 'stop_loss'
                            hit_stop_loss = True
                            break
                        elif trade_position == 'short' and current_price >= stop_loss_price:
                            exit_reason = 'stop_loss'
                            hit_stop_loss = True
                            break
                    
                    exit_idx += 1
                
                # Use the determined exit index
                if exit_idx > len(df) - 1:
                    exit_idx = len(df) - 1
                
                exit_date = df.index[exit_idx]
                
                # If stop loss hit, use stop loss price, otherwise use close
                if hit_stop_loss:
                    exit_price = stop_loss_price
                else:
                    exit_price = df.iloc[exit_idx]['Close']
                
                # Calculate P&L based on actual position quantity
                if trade_position == 'long':
                    pnl = (exit_price - entry_price) * position_quantity
                    points_captured = exit_price - entry_price
                else:  # short
                    pnl = (entry_price - exit_price) * position_quantity
                    points_captured = entry_price - exit_price
                
                pnl_percent = (pnl / (entry_price * position_quantity)) * 100
                
                trade = Trade(
                    symbol=symbol,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position=trade_position,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    points_captured=points_captured,
                    exit_reason=exit_reason,
                    stop_loss_price=stop_loss_price,
                    atr_value=atr_value,
                    risk_amount=actual_risk_amount,
                    position_quantity=position_quantity
                )
                trades.append(trade)
                
                # Skip forward to avoid overlapping trades
                i = exit_idx + 1
            else:
                i += 1
        
        return trades
    
    def _calculate_metrics(self, trades: List[Trade]) -> BacktestResult:
        """Calculate performance metrics from trades"""
        if not trades:
            return BacktestResult(
                trades=[], total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, total_points_captured=0, avg_points_per_trade=0,
                avg_win=0, avg_loss=0, avg_pnl=0, max_win=0, max_loss=0, profit_factor=0,
                stop_loss_hits=0, avg_risk_per_trade=0, total_risk_taken=0, 
                risk_reward_ratio=0, symbol_data={}
            )
        
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        stop_loss_trades = [t for t in trades if t.exit_reason == 'stop_loss']
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        total_points = sum(t.points_captured for t in trades)
        
        # Risk analysis based on actual risk amounts
        total_risk = sum(t.risk_amount for t in trades)
        avg_risk = total_risk / len(trades) if trades else 0
        
        # Risk/Reward ratio
        avg_reward = total_wins / len(winning_trades) if winning_trades else 0
        avg_risk_amount = total_losses / len(losing_trades) if losing_trades else 1
        risk_reward = avg_reward / avg_risk_amount if avg_risk_amount > 0 else 0
        
        return BacktestResult(
            trades=trades,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=(len(winning_trades) / len(trades)) * 100 if trades else 0,
            total_pnl=sum(t.pnl for t in trades),
            total_points_captured=total_points,
            avg_points_per_trade=total_points / len(trades),
            avg_win=total_wins / len(winning_trades) if winning_trades else 0,
            avg_loss=total_losses / len(losing_trades) if losing_trades else 0,
            avg_pnl=sum(t.pnl for t in trades) / len(trades),
            max_win=max((t.pnl for t in trades), default=0),
            max_loss=min((t.pnl for t in trades), default=0),
            profit_factor=total_wins / total_losses if total_losses > 0 else 0,
            stop_loss_hits=len(stop_loss_trades),
            avg_risk_per_trade=avg_risk,
            total_risk_taken=total_risk,
            risk_reward_ratio=risk_reward,
            symbol_data={}
        )

# =============================================================================
# EXAMPLE STRATEGY
# =============================================================================

def example_strategy(df: pd.DataFrame, symbol: str) -> dict:
    """
    STRATEGY: Long Only PDC-ORB
    - Entry: Breakout above Max(PDC, 1st Candle High)
    - SL: Low of Previous Day's Last Candle
    - Exit: Time-based (3 Candles) handled by Backtester config
    """
    
    # Need enough data to find previous day
    if len(df) < 25: return {}
    
    curr_timestamp = df.index[-1]
    curr_date = curr_timestamp.date()
    
    # 1. Get Data Groups
    today_data = df[df.index.date == curr_date]
    prev_data = df[df.index.date < curr_date]
    
    # Must have previous day data and at least the 1st candle of today closed
    if prev_data.empty or len(today_data) < 2:
        return {}

    # 2. Define Key Levels
    
    # A. Previous Day Close (PDC)
    pdc = prev_data.iloc[-1]['Close']
    
    # B. Previous Day Last Candle Low (Your Specific SL)
    prev_day_last_low = prev_data.iloc[-1]['Low']
    
    # C. First 15-Min Candle Stats
    orb_candle = today_data.iloc[0]
    orb_high = orb_candle['High'] # Using High for safe breakout
    
    # 3. Define Breakout Level (Upper Boundary)
    # We only care about the upper side since it's LONG ONLY
    breakout_level = max(pdc, orb_high)
    
    # 4. Validity Filter ("Not far from each other")
    # If the distance between PDC and ORB High is huge, it's a risky gap
    gap_pct = abs(orb_high - pdc) / pdc * 100
    if gap_pct > 1.5: # 1.5% threshold for "too far"
        return {}

    # 5. Check Trigger on Current Candle
    curr_close = df.iloc[-1]['Close']
    
    # LONG ENTRY LOGIC
    if curr_close > breakout_level:
        
        # Stop Loss Validation
        # If the gap up is huge, your SL (Yesterday's Low) might be too far or above price.
        # We only take trade if SL is logically below our entry.
        if prev_day_last_low >= curr_close:
            return {} # Invalid trade (SL is higher than Entry)

        return {
            'signal': True,
            'position': 'long',
            'sl_price': prev_day_last_low, # Explicit SL as requested
            'target_price': curr_close * 1.5 # Dummy high target (Exit is controlled by Time)
        }
        
    return {'signal': False}
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize Angel Broking API
    api = AngelBrokingAPI()
    
    # Initialize backtester
    backtester = Backtester(api)
    
    # NIFTY 50 STOCKS - All top 50 stocks

    # Data can be fetch from yfinance too if  needed

    nifty50_symbols = [
        # Banking & Finance (16 stocks)
        'HDFCBANK-EQ', 'ICICIBANK-EQ', 'SBIN-EQ', 'KOTAKBANK-EQ', 'AXISBANK-EQ',
        'BAJFINANCE-EQ', 'BAJAJFINSV-EQ', 'INDUSINDBK-EQ', 'HDFCLIFE-EQ', 
        'SBILIFE-EQ', 'LICI-EQ', 'JIOFIN-EQ', 'SHRIRAMFIN-EQ', 'BAJAJ-AUTO-EQ',
        'HDFCAMC-EQ', 'POONAWALLA-EQ',
        
        # IT (5 stocks)
        'TCS-EQ', 'INFY-EQ', 'WIPRO-EQ', 'HCLTECH-EQ', 'TECHM-EQ',
        
        # Energy & Utilities (6 stocks)
        'RELIANCE-EQ', 'ONGC-EQ', 'POWERGRID-EQ', 'COALINDIA-EQ', 
        'NTPC-EQ', 'BPCL-EQ',
        
        # Consumer Goods (5 stocks)
        'HINDUNILVR-EQ', 'ITC-EQ', 'NESTLEIND-EQ', 'BRITANNIA-EQ', 
        'DABUR-EQ',
        
        # Auto (5 stocks)
        'MARUTI-EQ', 'M&M-EQ', 'TATAMOTORS-EQ', 'EICHERMOT-EQ', 
        'HEROMOTOCO-EQ',
        
        # Pharma (4 stocks)
        'SUNPHARMA-EQ', 'DRREDDY-EQ', 'CIPLA-EQ', 'DIVISLAB-EQ',
        
        # Telecom & Tech (2 stocks)
        'BHARTIARTL-EQ', 'LT-EQ',
        
        # Metals & Mining (3 stocks)
        'HINDALCO-EQ', 'TATASTEEL-EQ', 'JSWSTEEL-EQ',
        
        # Others (4 stocks)
        'ADANIENT-EQ', 'ADANIPORTS-EQ', 'ULTRACEMCO-EQ', 'GRASIM-EQ'
    ]
    
    # For quick testing, start with top 10
    top10_symbols = [
        'RELIANCE-EQ', 'TCS-EQ', 'HDFCBANK-EQ', 'INFY-EQ', 
        'ICICIBANK-EQ', 'HINDUNILVR-EQ', 'ITC-EQ', 
        'SBIN-EQ', 'BHARTIARTL-EQ', 'KOTAKBANK-EQ'
    ]
    
    # Choose which list to use
    symbols_to_test = top10_symbols  # Change to nifty50_symbols for full test
    
    print(f"\n{'='*70}")    
    print(f"BACKTESTING STRATEGY ON {len(symbols_to_test)} STOCKS")
    print(f"{'='*70}")
    
    # Run backtest
    results = backtester.run_backtest(
        strategy_func=example_strategy,
        symbols=symbols_to_test,
        lookback_days=60,  # 1 year backtest
        holding_period_days=1,
        position_type='long',  # Strategy decides: 'long' for golden cross, 'short' for death cross
        initial_capital=100000,  # ₹1 Lakh
        position_size=0.1  # 10% of capital per trade (only used when stop loss disabled)
    )
    
    # Print comprehensive results with yearly and per-stock breakdown
    results.print_summary(show_yearly=True, show_per_stock=True, show_risk_analysis=True)
    
    # Get trades as DataFrame for further analysis
    trades_df = results.get_trades_df()
    if not trades_df.empty:
        print("\nTop 10 Best Trades:")
        top_trades = trades_df.nlargest(10, 'pnl_percent')[['symbol', 'entry_date', 'position', 'points_captured', 'pnl_percent', 'exit_reason']]
        print(top_trades.to_string(index=False))
        
        print("\nTop 10 Worst Trades:")
        worst_trades = trades_df.nsmallest(10, 'pnl_percent')[['symbol', 'entry_date', 'position', 'points_captured', 'pnl_percent', 'exit_reason']]
        print(worst_trades.to_string(index=False))
        
        # Plot trades
        print("\n" + "="*60)
        print("TRADE VISUALIZATION")
        print("="*60)
        results.plot_trades(max_plots=3)  # Plot first 3 stocks with trades
        
        # Option to test on Nifty 50
        print("\n" + "="*60)
        user_input = input("Do you want to run the same strategy on NIFTY 50? (yes/no): ").strip().lower()
        
        if user_input in ['yes', 'y']:
            print("\n" + "="*70)
            print("RUNNING BACKTEST ON NIFTY 50 INDEX")
            print("="*70)
            
            # Run on Nifty 50
            results_nifty = backtester.run_backtest(
                strategy_func=example_strategy,
                symbols=nifty50_symbols,
                lookback_days=365,
                holding_period_days=3,
                position_type='long',
                initial_capital=100000,
                position_size=0.1
            )
            
            print("\n" + "="*60)
            print("NIFTY 50 RESULTS")
            print("="*60)
            results_nifty.print_summary(show_yearly=True, show_per_stock=True, show_risk_analysis=True)
            
            # Compare both results
            print("\n" + "="*60)
            print("COMPARISON: YOUR SELECTION vs NIFTY 50")
            print("="*60)
            print(f"{'Metric':<30} {'Your Selection':<20} {'Nifty 50':<20}")
            print("-" * 70)
            print(f"{'Total Trades':<30} {results.total_trades:<20} {results_nifty.total_trades:<20}")
            print(f"{'Win Rate':<30} {results.win_rate:<20.2f}% {results_nifty.win_rate:<20.2f}%")
            print(f"{'Total P&L':<30} ₹{results.total_pnl:<19.2f} ₹{results_nifty.total_pnl:<19.2f}")
            print(f"{'Avg P&L per Trade':<30} ₹{results.avg_pnl:<19.2f} ₹{results_nifty.avg_pnl:<19.2f}")
            print(f"{'Total Points Captured':<30} {results.total_points_captured:<20.2f} {results_nifty.total_points_captured:<20.2f}")
            print(f"{'Profit Factor':<30} {results.profit_factor:<20.2f} {results_nifty.profit_factor:<20.2f}")
            print(f"{'Stop Loss Hits':<30} {results.stop_loss_hits:<20} {results_nifty.stop_loss_hits:<20}")
            print(f"{'Risk/Reward Ratio':<30} 1:{results.risk_reward_ratio:<18.2f} 1:{results_nifty.risk_reward_ratio:<18.2f}")
            print("="*70)
            
            # Recommendation
            if results.total_pnl > results_nifty.total_pnl:
                print("\n✅ Your selection performed BETTER than Nifty 50!")
            elif results.total_pnl < results_nifty.total_pnl:
                print("\n⚠️ Nifty 50 performed BETTER than your selection")
            else:
                print("\n➡️ Both performed equally")
        else:
            print("\nSkipping Nifty 50 comparison.")