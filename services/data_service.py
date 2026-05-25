import yfinance as yf
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_stock_data(ticker, period="6mo", interval="1h"):
    """
    Fetches historical data from Yahoo Finance with robust error handling.
    """
    try:
        logger.info(f"Fetching data for {ticker} (period={period}, interval={interval})")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return None, None
            
        currency = stock.info.get('currency', 'USD')
        df.reset_index(inplace=True)
        
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
            
        return df, currency
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None, None

def calculate_atr(df, period=14):
    """
    Calculates Average True Range.
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    return true_range.rolling(period).mean()

def add_technical_indicators(df):
    """
    Adds advanced technical indicators with proper NaN handling and feature engineering.
    """
    if df is None or df.empty:
        return None

    # 1. Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 2. RSI (14 period)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 4. Bollinger Bands
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (std_20 * 2)
    df['BB_Lower'] = df['SMA_20'] - (std_20 * 2)
    
    # 5. Volatility (ATR)
    df['ATR'] = calculate_atr(df)
    
    # --- PRO IMPROVEMENTS ---
    
    # 6. Returns (pct_change) - Crucial for ML
    df['Returns'] = df['Close'].pct_change()
    
    # 7. Rolling Volatility (20-period standard deviation of returns)
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # 8. Momentum (ROC - Rate of Change)
    df['Momentum'] = (df['Close'] / df['Close'].shift(10)) - 1
    
    # 9. Lagged Returns (capturing autocorrelation)
    df['Prev_Returns'] = df['Returns'].shift(1)
    
    # Drop initial rows with NaN values from rolling calculations
    # We use dropna() but keep it careful to not leak future info
    df.dropna(inplace=True)
    
    return df
