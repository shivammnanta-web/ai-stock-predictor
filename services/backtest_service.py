"""
backtest_service.py — Simulated trading engine for evaluating model performance.

Provides:
  - Walk-forward backtesting (train on past, predict future, step forward)
  - Portfolio tracking (equity curve, PnL, win rate)
  - Trade-by-trade history for the frontend
  - JSON-serializable results for the /api/backtest endpoint
"""

import numpy as np
import pandas as pd
from model import StockPredictor
from utils.logger import get_logger

logger = get_logger(__name__)


def run_backtest(df, initial_capital=10000.0, commission_pct=0.001):
    """
    Walk-forward backtest using the StockPredictor.

    Strategy:
      - Train on the first 60 % of data
      - Walk through the remaining 40 % bar-by-bar
      - At each bar the model predicts BUY / SELL / HOLD
      - BUY  → go long  (invest 100 % of available cash)
      - SELL → close any long position
      - HOLD → do nothing

    Args:
        df:               DataFrame with technical indicators already added
        initial_capital:   Starting cash (default $10 000)
        commission_pct:    Round-trip commission as a fraction (default 0.1 %)

    Returns:
        dict with keys: summary, trades, equity_curve
    """
    predictor = StockPredictor()
    features = predictor.features

    # Validate that all required features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in DataFrame: {missing}")

    # ── Prepare target (same logic as model.py) ──────────────────────
    df = df.copy()
    df['Future_Return'] = df['Close'].pct_change().shift(-1)
    df['Target'] = np.where(
        df['Future_Return'] > predictor.THRESHOLD, 1,
        np.where(df['Future_Return'] < -predictor.THRESHOLD, 0, np.nan)
    )
    df.dropna(subset=['Target'] + features, inplace=True)
    df['Target'] = df['Target'].astype(int)

    if len(df) < 100:
        logger.warning("Not enough data for backtesting")
        return {"error": "Insufficient data for backtesting (need ≥100 rows)"}

    # ── Split: 60 % initial train, walk forward on 40 % ────────────
    split_idx = int(len(df) * 0.6)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Train the model on the initial training window
    X_train = train_df[features]
    y_train = train_df['Target']
    predictor.model = predictor._get_candidate_models()['RandomForest']
    predictor.model.fit(X_train, y_train)
    predictor.model_name = 'RandomForest'

    # ── Walk-forward simulation ──────────────────────────────────────
    cash = initial_capital
    shares = 0
    position_open = False
    entry_price = 0.0

    trades = []
    equity_curve = []

    for i, (idx, row) in enumerate(test_df.iterrows()):
        price = float(row['Close'])
        X_row = row[features].values.reshape(1, -1)

        try:
            prob = predictor.model.predict_proba(X_row)[0]
            buy_prob = prob[1] if len(prob) > 1 else 0.5
        except Exception:
            buy_prob = 0.5

        # ── Decision logic ──
        if buy_prob > 0.55 and not position_open:
            # BUY
            commission = cash * commission_pct
            shares = (cash - commission) / price
            entry_price = price
            cash = 0.0
            position_open = True

            trades.append({
                'type': 'BUY',
                'price': round(price, 2),
                'confidence': round(buy_prob * 100, 2),
                'date': str(row.get('Date', f'bar_{i}')),
            })

        elif buy_prob < 0.45 and position_open:
            # SELL
            proceeds = shares * price
            commission = proceeds * commission_pct
            cash = proceeds - commission
            pnl = round(((price - entry_price) / entry_price) * 100, 2)

            trades.append({
                'type': 'SELL',
                'price': round(price, 2),
                'confidence': round((1 - buy_prob) * 100, 2),
                'pnl_pct': pnl,
                'date': str(row.get('Date', f'bar_{i}')),
            })

            shares = 0
            position_open = False

        # ── Track portfolio value ──
        portfolio_value = cash + (shares * price)
        equity_curve.append({
            'date': str(row.get('Date', f'bar_{i}')),
            'value': round(portfolio_value, 2),
        })

    # ── Close any open position at the end ──
    if position_open:
        final_price = float(test_df.iloc[-1]['Close'])
        proceeds = shares * final_price
        cash = proceeds - (proceeds * commission_pct)
        pnl = round(((final_price - entry_price) / entry_price) * 100, 2)
        trades.append({
            'type': 'SELL (forced close)',
            'price': round(final_price, 2),
            'pnl_pct': pnl,
            'date': str(test_df.iloc[-1].get('Date', 'last')),
        })

    # ── Compute summary statistics ───────────────────────────────────
    final_value = cash if not position_open else cash + shares * float(test_df.iloc[-1]['Close'])

    sell_trades = [t for t in trades if 'pnl_pct' in t]
    wins = [t for t in sell_trades if t['pnl_pct'] > 0]
    losses = [t for t in sell_trades if t['pnl_pct'] <= 0]

    total_return_pct = round(((final_value - initial_capital) / initial_capital) * 100, 2)
    win_rate = round(len(wins) / len(sell_trades) * 100, 1) if sell_trades else 0

    # Buy-and-hold benchmark
    bh_start = float(test_df.iloc[0]['Close'])
    bh_end = float(test_df.iloc[-1]['Close'])
    buy_hold_return = round(((bh_end - bh_start) / bh_start) * 100, 2)

    summary = {
        'initial_capital': initial_capital,
        'final_value': round(final_value, 2),
        'total_return_pct': total_return_pct,
        'buy_hold_return_pct': buy_hold_return,
        'alpha_pct': round(total_return_pct - buy_hold_return, 2),
        'total_trades': len(trades),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': win_rate,
        'avg_win_pct': round(np.mean([t['pnl_pct'] for t in wins]), 2) if wins else 0,
        'avg_loss_pct': round(np.mean([t['pnl_pct'] for t in losses]), 2) if losses else 0,
        'test_period_bars': len(test_df),
    }

    logger.info(
        f"Backtest complete: Return={total_return_pct}% "
        f"(B&H={buy_hold_return}%) Win rate={win_rate}% "
        f"over {len(trades)} trades"
    )

    return {
        'summary': summary,
        'trades': trades,
        'equity_curve': equity_curve,  # Subsample for large datasets
    }
