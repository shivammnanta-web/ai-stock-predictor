"""
index.py — Vercel-compatible deployment entry point.

Improvements over original:
  1. Smarter rule-based fallback with dynamic confidence and explanations
  2. Graceful degradation: ML failure → rule-based → minimal response
  3. Proper structured logging instead of print()
  4. Consistent error responses matching app.py
  5. NLTK path handling for serverless environments
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import numpy as np
from datetime import datetime, timedelta

# ── Logging (works even if utils/ fails to import) ───────────────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('index')

# ── Import with multi-level fallback ─────────────────────────────────
INIT_ERROR = None
USE_ML_FALLBACK = False
HAS_BACKTEST = False

try:
    from utils.json_encoder import NumpyEncoder
except ImportError:
    # Inline fallback if utils package doesn't load on Vercel
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

try:
    from data_fetcher import fetch_stock_data, add_technical_indicators
    from feedback_store import (
        save_feedback_validated, load_feedback,
        get_feedback_for_ticker, get_feedback_summary,
        get_accuracy_trend, get_per_ticker_stats
    )
    from news_analyzer import NewsAnalyzer
except ImportError as e:
    INIT_ERROR = f"Core import error: {e}"
    USE_ML_FALLBACK = True
    logger.error(INIT_ERROR)

try:
    from model import StockPredictor
except ImportError as e:
    logger.warning(f"ML model unavailable: {e} — using rule-based fallback")
    USE_ML_FALLBACK = True

try:
    from services.backtest_service import run_backtest
    HAS_BACKTEST = True
except ImportError:
    logger.info("Backtest service not available in this environment")

# ── NLTK data paths ──────────────────────────────────────────────────
try:
    import nltk
    local_nltk_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
    tmp_nltk_path = '/tmp/nltk_data'
    for p in [local_nltk_path, tmp_nltk_path]:
        if os.path.exists(p) and p not in nltk.data.path:
            nltk.data.path.append(p)
except Exception:
    pass

# ── Flask app ────────────────────────────────────────────────────────
try:
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
except ImportError:
    pass

app = Flask(__name__)
CORS(app)
try:
    app.json = CustomJSONProvider(app)
except NameError:
    app.json.cls = NumpyEncoder

# ── Global model cache ───────────────────────────────────────────────
_model_cache = {}


# ══════════════════════════════════════════════════════════════════════
# SMART RULE-BASED FALLBACK
# ══════════════════════════════════════════════════════════════════════

def predict_rule_based(df):
    """
    Intelligent rule-based predictor used when ML stack is unavailable.

    Uses a multi-indicator scoring approach instead of a single
    if/else chain. Each indicator contributes to a cumulative score,
    producing a dynamic confidence value with an explanation.
    """
    latest = df.iloc[-1]
    close = float(latest['Close'])
    rsi = float(latest['RSI'])
    sma20 = float(latest['SMA_20'])
    sma50 = float(latest['SMA_50'])
    atr = float(latest['ATR'])
    macd = float(latest['MACD'])
    macd_signal = float(latest['MACD_Signal'])

    score = 0       # −5 (strong sell) to +5 (strong buy)
    reasons = []

    # ── RSI ──
    if rsi < 30:
        score += 2
        reasons.append(f"RSI oversold ({rsi:.1f})")
    elif rsi < 40:
        score += 1
        reasons.append(f"RSI approaching oversold ({rsi:.1f})")
    elif rsi > 70:
        score -= 2
        reasons.append(f"RSI overbought ({rsi:.1f})")
    elif rsi > 60:
        score -= 1
        reasons.append(f"RSI elevated ({rsi:.1f})")

    # ── Price vs SMA 20 ──
    if close > sma20 * 1.02:
        score += 1
        reasons.append("Price above SMA20 (bullish)")
    elif close < sma20 * 0.98:
        score -= 1
        reasons.append("Price below SMA20 (bearish)")

    # ── SMA 20 vs SMA 50 (Golden/Death cross) ──
    if sma20 > sma50:
        score += 1
        reasons.append("SMA20 > SMA50 (uptrend)")
    else:
        score -= 1
        reasons.append("SMA20 < SMA50 (downtrend)")

    # ── MACD ──
    if macd > macd_signal:
        score += 1
        reasons.append("MACD bullish crossover")
    else:
        score -= 1
        reasons.append("MACD bearish crossover")

    # ── Decision ──
    max_score = 5
    confidence = 50 + (abs(score) / max_score) * 35  # 50–85% range

    if score >= 2:
        action = "BUY"
        stop_loss = close - (1.5 * atr)
    elif score <= -2:
        action = "SELL"
        stop_loss = close + (1.5 * atr)
    else:
        action = "HOLD"
        stop_loss = None
        confidence = 50 + (abs(score) / max_score) * 15  # Lower for HOLD

    explanation = (
        f"Rule-based analysis (ML unavailable). "
        f"Score: {score:+d}/{max_score}. "
        + "; ".join(reasons[:3])
    )

    return {
        "action": action,
        "confidence": round(confidence, 2),
        "stop_loss": round(stop_loss, 2) if stop_loss else None,
        "current_price": round(close, 2),
        "buy_prob": round(0.5 + (score / (max_score * 2)), 3),
        "explanation": explanation,
        "model_used": "RuleBased",
        "is_fallback": True,
    }


# ── Error helper ─────────────────────────────────────────────────────

def error_response(message, status_code=500, details=None):
    payload = {'error': True, 'message': message, 'status': status_code}
    if details:
        payload['details'] = details
    return jsonify(payload), status_code


# ══════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    status = "degraded" if USE_ML_FALLBACK else "healthy"
    return jsonify({
        'status': status,
        'ml_available': not USE_ML_FALLBACK,
        'backtest_available': HAS_BACKTEST,
        'error': INIT_ERROR,
        'vercel': bool(os.environ.get('VERCEL')),
    })


@app.route('/api/analyze/<ticker>', methods=['GET'])
def analyze_ticker(ticker):
    try:
        ticker = ticker.upper()
        logger.info(f"Analyzing {ticker}")

        df, currency = fetch_stock_data(ticker)
        if df is None or df.empty:
            return error_response(f'Could not fetch data for {ticker}', 404)

        df = add_technical_indicators(df)
        if df is None or df.empty:
            return error_response(f'Not enough data for {ticker}', 400)

        # ── Prediction (ML with fallback) ──
        eval_summary = {}
        if not USE_ML_FALLBACK:
            try:
                # Try cache → disk → train
                if ticker in _model_cache:
                    predictor = _model_cache[ticker]
                else:
                    predictor = StockPredictor()
                    if not predictor.load_model(ticker):
                        predictor.train_and_evaluate(df)
                        predictor.save_model(ticker)
                    _model_cache[ticker] = predictor

                prediction = predictor.predict_latest(df)
                eval_summary = predictor.get_evaluation_summary()
            except Exception as e:
                logger.warning(f"ML failed for {ticker}, using fallback: {e}")
                prediction = predict_rule_based(df)
        else:
            prediction = predict_rule_based(df)

        # ── Sentiment ──
        try:
            news_analyzer = NewsAnalyzer()
            news_items = news_analyzer.get_stock_news(ticker)
            sentiment = news_analyzer.analyze_sentiment(news_items)
            diplomatic = news_analyzer.diplomatic_fusion(
                prediction.get('buy_prob', 0.5),
                sentiment['average_sentiment']
            )
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            sentiment = {"average_sentiment": 0, "overall_tone": "Neutral", "headlines": []}
            diplomatic = {"diplomatic_action": prediction['action'],
                          "fused_score": prediction['confidence'],
                          "reasoning": "Sentiment unavailable", "weights": {}}

        # ── Chart data ──
        chart_df = df.tail(100)
        timestamps = chart_df['Date'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        prices = chart_df['Close'].tolist()

        last_time = df['Date'].iloc[-1]
        next_time = last_time + timedelta(hours=1)
        predicted_time_str = next_time.strftime('%A, %b %d at %I:%M %p')

        # ── Feature snapshot ──
        feature_cols = (
            predictor.features if not USE_ML_FALLBACK and 'predictor' in dir()
            else ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',
                   'BB_Upper', 'BB_Lower', 'ATR']
        )
        available_cols = [c for c in feature_cols if c in df.columns]
        latest_features = df[available_cols].iloc[-1].to_dict()

        # ── Community stats ──
        try:
            ticker_fb = get_feedback_for_ticker(ticker)
            community_correct = sum(1 for e in ticker_fb if e.get('did_it_work'))
            community_total = len(ticker_fb)
            community_accuracy = (
                round(community_correct / community_total * 100, 1)
                if community_total > 0 else None
            )
        except Exception:
            community_total, community_correct, community_accuracy = 0, 0, None

        response = {
            'ticker': ticker,
            'current_price': prediction['current_price'],
            'currency': currency,
            'action': prediction['action'],
            'confidence': prediction['confidence'],
            'stop_loss': prediction['stop_loss'],
            'success_rate': eval_summary.get('metrics', {}).get('precision', 0),
            'predicted_time': predicted_time_str,
            'explanation': prediction.get('explanation', ''),
            'is_fallback': prediction.get('is_fallback', False),
            'latest_features': latest_features,
            'model_info': {
                'model_used': prediction.get('model_used', 'Unknown'),
                'metrics': eval_summary.get('metrics', {}),
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
                'headlines': sentiment.get('headlines', [])[:5],
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

        logger.info(f"Analysis complete for {ticker}: {prediction['action']}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)
        return error_response('An internal error occurred during analysis.')


# ── Backtesting ──────────────────────────────────────────────────────

@app.route('/api/backtest/<ticker>', methods=['GET'])
def backtest_ticker(ticker):
    if not HAS_BACKTEST:
        return error_response('Backtesting not available in this environment', 501)

    try:
        ticker = ticker.upper()
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
        logger.error(f"Backtest error: {e}", exc_info=True)
        return error_response('Backtest failed.')


# ── Feedback ─────────────────────────────────────────────────────────

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
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


# ── Retrain ──────────────────────────────────────────────────────────

@app.route('/api/retrain/<ticker>', methods=['POST'])
def retrain_ticker(ticker):
    if USE_ML_FALLBACK:
        return error_response('ML model not available for retraining', 501)

    try:
        ticker = ticker.upper()
        df, currency = fetch_stock_data(ticker)
        if df is None or df.empty:
            return error_response(f'Could not fetch data for {ticker}', 404)

        df = add_technical_indicators(df)
        if df is None or df.empty:
            return error_response('Not enough data', 400)

        predictor = StockPredictor()

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
        _model_cache[ticker] = predictor

        prediction = predictor.predict_latest(df)

        return jsonify({
            'message': f'Model retrained for {ticker}.',
            'feedback_samples_used': len(feedback_samples),
            'updated_prediction': {
                'action': prediction['action'],
                'confidence': prediction['confidence'],
                'stop_loss': prediction['stop_loss'],
            },
        })

    except Exception as e:
        logger.error(f"Retrain error: {e}", exc_info=True)
        return error_response('Failed to retrain model.')


# ── Search ───────────────────────────────────────────────────────────

@app.route('/api/search', methods=['GET'])
def search_tickers():
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify({'results': []})
    try:
        import yfinance as yf
        search = yf.Search(query, max_results=8)
        results = [
            {'ticker': q.get('symbol'), 'name': q.get('shortname'), 'exchange': q.get('exchange')}
            for q in search.quotes
        ]
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'results': []}), 500


# ── Feedback analytics ───────────────────────────────────────────────

@app.route('/api/feedback/summary', methods=['GET'])
def feedback_summary_route():
    summary = get_feedback_summary()
    by_ticker = get_per_ticker_stats()
    return jsonify({'global': summary, 'by_ticker': by_ticker})


@app.route('/api/feedback/trend', methods=['GET'])
def feedback_trend_route():
    window = request.args.get('window', 10, type=int)
    trend = get_accuracy_trend(window=window)
    return jsonify({'trend': trend, 'window': window})


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    logger.info("Starting StockSense AI on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
