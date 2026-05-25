"""
data_fetcher.py — Stock data retrieval and feature engineering.

Improvements over original:
  1. Proper logging instead of print()
  2. fetch_stock_data always returns a tuple (df, currency) — never bare None
  3. New features: Returns, Volatility (rolling std), Momentum (ROC), Prev_Returns
  4. Proper NaN handling to prevent data leakage
  5. Null-safe add_technical_indicators (returns None on bad input)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


def fetch_stock_data(ticker, period="6mo", interval="1h"):
    """
    Fetches historical OHLCV data from Yahoo Finance.

    Args:
        ticker:   Stock symbol (e.g. "AAPL")
        period:   Lookback window (default "6mo")
        interval: Bar size (default "1h")

    Returns:
        (DataFrame, currency_str) on success
        (None, None) on failure — callers should always unpack both
    """
    try:
        logger.info(f"Fetching {ticker} data (period={period}, interval={interval})")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return None, None

        currency = stock.info.get('currency', 'USD')

        # Normalize index → 'Date' column
        df.reset_index(inplace=True)
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)

        logger.info(f"Fetched {len(df)} rows for {ticker}")
        return df, currency

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}", exc_info=True)
        return None, None


# ── Technical Indicator Helpers ──────────────────────────────────────

def calculate_atr(df, period=14):
    """Average True Range — volatility measure based on candle ranges."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    return true_range.rolling(period).mean()


def add_technical_indicators(df):
    """
    Enriches a price DataFrame with technical indicators.

    Added indicators:
      - SMA_20, SMA_50      (trend)
      - RSI                  (momentum oscillator)
      - MACD, MACD_Signal    (trend-following momentum)
      - BB_Upper, BB_Lower   (Bollinger Bands — volatility)
      - ATR                  (Average True Range)
      - Returns              (period-over-period % change)
      - Volatility           (20-period rolling std of returns)
      - Momentum             (10-period rate of change)
      - Prev_Returns         (lagged return for autocorrelation capture)

    NaN rows produced by rolling windows are dropped at the end
    so downstream code never sees them.
    """
    if df is None or df.empty:
        logger.warning("add_technical_indicators received empty input")
        return None

    df = df.copy()  # Never mutate the caller's DataFrame

    # ── Trend ────────────────────────────────────────────────
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # ── RSI (14) ─────────────────────────────────────────────
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────────────
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ── Bollinger Bands ──────────────────────────────────────
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (std_20 * 2)
    df['BB_Lower'] = df['SMA_20'] - (std_20 * 2)

    # ── ATR ──────────────────────────────────────────────────
    df['ATR'] = calculate_atr(df)

    # ── NEW: Returns & derived features ──────────────────────
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['Momentum'] = (df['Close'] / df['Close'].shift(10)) - 1
    df['Prev_Returns'] = df['Returns'].shift(1)

    # ── Drop NaN rows from rolling calculations ──────────────
    initial_len = len(df)
    df.dropna(inplace=True)
    logger.info(f"Indicators added: {initial_len} → {len(df)} rows (dropped {initial_len - len(df)} NaN rows)")

    return df


# ── Self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing data fetcher with AAPL...")
    result = fetch_stock_data("AAPL")
    if result[0] is not None:
        data, currency = result
        print(f"Fetched {len(data)} rows  (currency: {currency})")
        data = add_technical_indicators(data)
        print(f"After indicators: {len(data)} rows")
        print(data[['Date', 'Close', 'RSI', 'ATR', 'Returns', 'Volatility', 'Momentum']].tail())
