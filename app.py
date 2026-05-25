"""
app.py — Main Flask API for StockSense AI.

Improvements over original:
  1. Global model cache — loads once per ticker, not per request
  2. Full evaluation metrics and feature importance in /api/analyze response
  3. New /api/backtest/<ticker> endpoint
  4. Standardized JSON error responses
  5. Centralized logging (no more bare print())
  6. Uses validated feedback store
  7. New /api/feedback/trend endpoint for accuracy tracking
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import numpy as np
from datetime import datetime, timedelta

from utils.logger import get_logger
from utils.json_encoder import NumpyEncoder

logger = get_logger(__name__)

# ── Import core modules with error tracking for Vercel debugging ────
INIT_ERROR = None
try:
    from data_fetcher import fetch_stock_data, add_technical_indicators
    from model import StockPredictor
    from feedback_store import (
        save_feedback, save_feedback_validated, load_feedback,
        get_feedback_for_ticker, get_feedback_summary,
        get_accuracy_trend, get_per_ticker_stats
    )
    from news_analyzer import NewsAnalyzer
    from services.backtest_service import run_backtest
except Exception as e:
    import traceback
    INIT_ERROR = f"Import Error: {str(e)}\n{traceback.format_exc()}"
    logger.error(f"Startup import error: {INIT_ERROR}")

# ── NLTK data paths ─────────────────────────────────────────────────
import nltk
local_nltk_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
tmp_nltk_path = '/tmp/nltk_data'
for p in [local_nltk_path, tmp_nltk_path]:
    if os.path.exists(p) and p not in nltk.data.path:
        nltk.data.path.append(p)

from flask.json.provider import DefaultJSONProvider

class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__)
CORS(app)
app.json = CustomJSONProvider(app)

# ── Global model cache (key: ticker → StockPredictor) ────────────────
# Models are trained once and reused across requests.
# The /api/retrain endpoint refreshes a specific ticker's model.
_model_cache = {}


def _get_or_train_model(ticker, df):
    """
    Returns a trained StockPredictor for the given ticker.
    Uses in-memory cache first, then tries loading from disk,
    and finally trains fresh if neither is available.
    """
    ticker = ticker.upper()

    # 1. In-memory cache hit
    if ticker in _model_cache:
        logger.info(f"Model cache hit for {ticker}")
        return _model_cache[ticker]

    predictor = StockPredictor()

    # 2. Try loading from disk (joblib)
    if predictor.load_model(ticker):
        _model_cache[ticker] = predictor
        return predictor

    # 3. Train fresh
    logger.info(f"Training new model for {ticker}")
    predictor.train_and_evaluate(df)
    predictor.save_model(ticker)
    _model_cache[ticker] = predictor
    return predictor


# ── Error helper ─────────────────────────────────────────────────────

def error_response(message, status_code=500, details=None):
    """Standardized JSON error response."""
    payload = {
        'error': True,
        'message': message,
        'status': status_code,
    }
    if details:
        payload['details'] = details
    return jsonify(payload), status_code


# ══════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/debug')
def debug():
    return jsonify({
        'vercel': bool(os.environ.get('VERCEL')),
        'init_error': INIT_ERROR,
        'cached_models': list(_model_cache.keys()),
    })


# ── Main analysis endpoint ───────────────────────────────────────────

@app.route('/api/analyze/<ticker>', methods=['GET'])
def analyze_ticker(ticker):
    """
    Full stock analysis: ML prediction + sentiment + diplomatic fusion.
    Now includes model metrics, feature importance, and explanation.
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Analyzing {ticker}")

        # Fetch & prepare data
        df, currency = fetch_stock_data(ticker)
        if df is None or df.empty:
            return error_response(f'Could not fetch data for ticker {ticker}', 404)

        df = add_technical_indicators(df)
        if df is None or df.empty:
            return error_response(f'Not enough data to compute indicators for {ticker}', 400)

        # Get or train model (cached)
        predictor = _get_or_train_model(ticker, df)
        prediction = predictor.predict_latest(df)
        eval_summary = predictor.get_evaluation_summary()

        # News & sentiment
        news_analyzer = NewsAnalyzer()
        news_items = news_analyzer.get_stock_news(ticker)
        sentiment = news_analyzer.analyze_sentiment(news_items)

        # Diplomatic fusion
        diplomatic = news_analyzer.diplomatic_fusion(
            prediction['buy_prob'],
            sentiment['average_sentiment']
        )

        # Chart data (last 100 bars)
        chart_df = df.tail(100)
        timestamps = chart_df['Date'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        prices = chart_df['Close'].tolist()

        # Predicted time
        last_time = df['Date'].iloc[-1]
        next_time = last_time + timedelta(hours=1)
        predicted_time_str = next_time.strftime('%A, %b %d at %I:%M %p')

        # Latest feature snapshot (for feedback retraining)
        latest_features = df[predictor.features].iloc[-1].to_dict()

        # Community stats
        ticker_fb = get_feedback_for_ticker(ticker)
        community_correct = sum(1 for e in ticker_fb if e.get('did_it_work'))
        community_total = len(ticker_fb)
        community_accuracy = (
            round(community_correct / community_total * 100, 1)
            if community_total > 0 else None
        )

        response = {
            'ticker': ticker,
            'current_price': prediction['current_price'],
            'currency': currency,
            'action': prediction['action'],
            'confidence': prediction['confidence'],
            'stop_loss': prediction['stop_loss'],
            'success_rate': eval_summary['metrics'].get('precision', 0),
            'predicted_time': predicted_time_str,
            'explanation': prediction.get('explanation', ''),
            'latest_features': latest_features,

            # NEW: Model evaluation details
            'model_info': {
                'model_used': prediction.get('model_used', 'Unknown'),
                'metrics': eval_summary['metrics'],
                'model_comparison': eval_summary.get('model_comparison', {}),
                'feature_importances': eval_summary.get('feature_importances', {}),
            },

            'community': {
                'total': community_total,
                'correct': community_correct,
                'accuracy': community_accuracy,
            },
            'news_sentiment': {
                'average_score': sentiment['average_sentiment'],
                'tone': sentiment['overall_tone'],
                'headlines': sentiment['headlines'][:5],
                'news_count': sentiment.get('news_count', 0),
            },
            'diplomatic_signal': {
                'action': diplomatic['diplomatic_action'],
                'score': diplomatic['fused_score'],
                'reasoning': diplomatic['reasoning'],
                'weights': diplomatic.get('weights', {}),
            },
            'chart_data': {
                'labels': timestamps,
                'prices': prices,
            },
        }

        logger.info(f"Analysis complete for {ticker}: {prediction['action']} ({prediction['confidence']}%)")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)
        return error_response('An internal error occurred during analysis.')


# ── Backtesting endpoint (NEW) ───────────────────────────────────────

@app.route('/api/backtest/<ticker>', methods=['GET'])
def backtest_ticker(ticker):
    """
    Runs a walk-forward backtest on the given ticker.
    Returns: summary stats, trade history, equity curve.
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Running backtest for {ticker}")

        df, currency = fetch_stock_data(ticker)
        if df is None or df.empty:
            return error_response(f'Could not fetch data for {ticker}', 404)

        df = add_technical_indicators(df)
        if df is None or df.empty:
            return error_response(f'Not enough data for {ticker}', 400)

        result = run_backtest(df)

        if 'error' in result:
            return error_response(result['error'], 400)

        result['ticker'] = ticker
        result['currency'] = currency
        return jsonify(result)

    except Exception as e:
        logger.error(f"Backtest error for {ticker}: {e}", exc_info=True)
        return error_response('Backtest failed.')


# ── Feedback endpoint ────────────────────────────────────────────────

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit user feedback on a prediction.
    Uses validated save to prevent corruption.
    """
    try:
        data = request.get_json()
        if not data:
            return error_response('No JSON body provided', 400)

        count, err = save_feedback_validated(data)
        if err:
            return error_response(err, 400)

        summary = get_feedback_summary()
        return jsonify({
            'message': 'Feedback saved! Thank you for helping improve the model. 🙏',
            'total_feedback': count,
            'global_summary': summary,
        })

    except Exception as e:
        logger.error(f"Feedback error: {e}", exc_info=True)
        return error_response('Failed to save feedback.')


# ── Retrain endpoint ─────────────────────────────────────────────────

@app.route('/api/retrain/<ticker>', methods=['POST'])
def retrain_ticker(ticker):
    """
    Force-retrain the model for a ticker, incorporating user feedback.
    Clears the cache entry so the next /analyze uses the new model.
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Retraining model for {ticker}")

        df, currency = fetch_stock_data(ticker)
        if df is None or df.empty:
            return error_response(f'Could not fetch data for {ticker}', 404)

        df = add_technical_indicators(df)
        if df is None or df.empty:
            return error_response('Not enough data', 400)

        predictor = StockPredictor()

        # Build feedback samples from confirmed entries
        ticker_fb = get_feedback_for_ticker(ticker)
        feedback_samples = []
        for entry in ticker_fb:
            if entry.get('did_it_work') and entry.get('latest_features'):
                feat = entry['latest_features']
                label = 1 if entry.get('action') == 'BUY' else 0
                if all(k in feat for k in predictor.features):
                    feedback_samples.append({
                        'features': [feat[k] for k in predictor.features],
                        'label': label,
                    })

        predictor.train_and_evaluate(df, feedback_samples=feedback_samples or None)
        predictor.save_model(ticker)

        # Update cache
        _model_cache[ticker] = predictor

        prediction = predictor.predict_latest(df)
        summary = predictor.get_evaluation_summary()

        return jsonify({
            'message': f'Model retrained for {ticker} using {len(feedback_samples)} feedback sample(s).',
            'feedback_samples_used': len(feedback_samples),
            'new_success_rate': summary['metrics'].get('precision', 0),
            'model_info': summary,
            'updated_prediction': {
                'action': prediction['action'],
                'confidence': prediction['confidence'],
                'stop_loss': prediction['stop_loss'],
            },
        })

    except Exception as e:
        logger.error(f"Retrain error for {ticker}: {e}", exc_info=True)
        return error_response('Failed to retrain model.')


# ── Search endpoint ──────────────────────────────────────────────────

@app.route('/api/search', methods=['GET'])
def search_tickers():
    """Search for tickers via yfinance."""
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify({'results': []})

    try:
        import yfinance as yf
        search = yf.Search(query, max_results=8)
        results = [
            {
                'ticker': quote.get('symbol'),
                'name': quote.get('shortname'),
                'exchange': quote.get('exchange'),
            }
            for quote in search.quotes
        ]
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'results': []}), 500


# ── Feedback analytics endpoints ─────────────────────────────────────

@app.route('/api/feedback/summary', methods=['GET'])
def feedback_summary_route():
    """Global + per-ticker feedback stats."""
    summary = get_feedback_summary()
    by_ticker = get_per_ticker_stats()
    return jsonify({'global': summary, 'by_ticker': by_ticker})


@app.route('/api/feedback/trend', methods=['GET'])
def feedback_trend_route():
    """Rolling accuracy trend for charting."""
    window = request.args.get('window', 10, type=int)
    trend = get_accuracy_trend(window=window)
    return jsonify({'trend': trend, 'window': window})


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    logger.info("Starting StockSense AI on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
