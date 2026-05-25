"""
model.py — Production-grade ML prediction engine.

Improvements over original:
  1. Threshold-based target (>0.5% move) to reduce noise
  2. TimeSeriesSplit cross-validation for honest evaluation
  3. Model comparison: RandomForest vs XGBoost (if available) vs LogisticRegression
  4. Extended feature set with Returns, Momentum, Volatility
  5. Model persistence via joblib (save/load instead of retraining)
  6. Full evaluation metrics: Accuracy, Precision, Recall, F1
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Directory for persisted models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Try importing XGBoost; graceful fallback if not installed
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.info("XGBoost not installed — skipping in model comparison")


class StockPredictor:
    """
    Multi-model stock predictor with time-series aware evaluation
    and intelligent signal generation.
    """

    # Expanded feature set: original indicators + returns/momentum/volatility
    FEATURES = [
        'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',
        'BB_Upper', 'BB_Lower', 'ATR',
        'Returns', 'Volatility', 'Momentum', 'Prev_Returns'
    ]

    # Price-move threshold — only label a sample as 1 / 0 if the
    # next-period return exceeds ±0.5 %. Samples inside the dead zone
    # are dropped, which reduces label noise substantially.
    THRESHOLD = 0.005  # 0.5 %

    def __init__(self):
        self.model = None           # Will hold the best-performing model
        self.model_name = None      # Name of the best model
        self.features = self.FEATURES
        self.metrics = {}           # Last evaluation metrics
        self.feature_importances = {}

    # ------------------------------------------------------------------
    # Target preparation
    # ------------------------------------------------------------------
    def prepare_data(self, df):
        """
        Creates a threshold-based target variable.
        Samples where the next-period move is < THRESHOLD are dropped
        to eliminate noisy, ambiguous labels.
        """
        df = df.copy()

        # Future return for the next period
        df['Future_Return'] = df['Close'].pct_change().shift(-1)

        # Threshold-based label
        df['Target'] = np.where(
            df['Future_Return'] > self.THRESHOLD, 1,
            np.where(df['Future_Return'] < -self.THRESHOLD, 0, np.nan)
        )

        # Drop ambiguous (dead-zone) and NaN rows
        df.dropna(subset=['Target'] + self.features, inplace=True)
        df['Target'] = df['Target'].astype(int)

        X = df[self.features]
        y = df['Target']

        return X, y, df

    # ------------------------------------------------------------------
    # Candidate models
    # ------------------------------------------------------------------
    def _get_candidate_models(self):
        """Returns a dict of name→estimator to compare."""
        candidates = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                random_state=42, n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000, random_state=42
            ),
        }
        if HAS_XGBOOST:
            candidates['XGBoost'] = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric='logloss',
                random_state=42, verbosity=0
            )
        return candidates

    # ------------------------------------------------------------------
    # Time-series cross-validation
    # ------------------------------------------------------------------
    def train_and_evaluate(self, df, feedback_samples=None, n_splits=5):
        """
        Evaluates all candidate models via TimeSeriesSplit,
        selects the best one, then trains it on the full training set.

        Returns:
            float — best model's average precision (%), kept for API compat
        """
        X, y, df_clean = self.prepare_data(df)

        if len(X) < 100:
            logger.warning(f"Only {len(X)} samples after threshold filtering — too few for reliable eval")
            # Fall back to simple RF without CV
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            self.model_name = 'RandomForest'
            self.metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            return 0.0

        tscv = TimeSeriesSplit(n_splits=min(n_splits, len(X) // 50))
        candidates = self._get_candidate_models()

        best_name, best_f1 = None, -1
        all_results = {}

        for name, estimator in candidates.items():
            fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Inject feedback samples into training set if available
                if feedback_samples:
                    X_train, y_train = self._inject_feedback(
                        X_train, y_train, feedback_samples
                    )

                try:
                    estimator.fit(X_train, y_train)
                    preds = estimator.predict(X_test)

                    fold_metrics['accuracy'].append(accuracy_score(y_test, preds))
                    fold_metrics['precision'].append(
                        precision_score(y_test, preds, zero_division=0)
                    )
                    fold_metrics['recall'].append(
                        recall_score(y_test, preds, zero_division=0)
                    )
                    fold_metrics['f1'].append(
                        f1_score(y_test, preds, zero_division=0)
                    )
                except Exception as e:
                    logger.error(f"Fold error for {name}: {e}")
                    continue

            avg = {k: round(np.mean(v) * 100, 2) if v else 0 for k, v in fold_metrics.items()}
            all_results[name] = avg
            logger.info(f"  {name:20s} → Acc={avg['accuracy']:.1f}%  Prec={avg['precision']:.1f}%  F1={avg['f1']:.1f}%")

            if avg['f1'] > best_f1:
                best_f1 = avg['f1']
                best_name = name

        # ---- Final training on full dataset with the best model ----
        self.model_name = best_name
        self.model = candidates[best_name]
        self.metrics = all_results.get(best_name, {})

        # Store comparison results so API can expose them
        self.model_comparison = all_results

        # Train on all data
        X_full, y_full = X, y
        if feedback_samples:
            X_full, y_full = self._inject_feedback(X_full, y_full, feedback_samples)
        self.model.fit(X_full, y_full)

        # Feature importances (tree-based models only)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.feature_importances = dict(
                sorted(
                    zip(self.features, [round(float(v), 4) for v in importances]),
                    key=lambda x: x[1], reverse=True
                )
            )

        logger.info(f"✅ Best model: {best_name} (F1={best_f1:.1f}%)")
        return round(self.metrics.get('precision', 0), 2)

    # ------------------------------------------------------------------
    # Feedback injection helper
    # ------------------------------------------------------------------
    def _inject_feedback(self, X_train, y_train, feedback_samples):
        """Injects user-confirmed feedback with 5× weight via duplication."""
        extra_X, extra_y = [], []
        for sample in feedback_samples:
            try:
                extra_X.append(sample['features'])
                extra_y.append(sample['label'])
            except KeyError:
                continue

        if extra_X:
            extra_df = pd.DataFrame(extra_X, columns=self.features)
            extra_s = pd.Series(extra_y)
            # Duplicate 5× for emphasis
            for _ in range(5):
                X_train = pd.concat([X_train, extra_df], ignore_index=True)
                y_train = pd.concat([y_train, extra_s], ignore_index=True)

        return X_train, y_train

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_latest(self, df):
        """
        Predicts the action for the latest data point.
        Returns action, confidence, stop-loss, and an explanation.
        """
        if self.model is None:
            raise RuntimeError("Model not trained — call train_and_evaluate() first")

        latest = df.iloc[-1:]
        X_latest = latest[self.features]
        prob = self.model.predict_proba(X_latest)[0]

        # Gracefully handle single-class models
        buy_prob = prob[1] if len(prob) > 1 else prob[0]

        current_price = float(latest['Close'].values[0])
        current_atr = float(latest['ATR'].values[0])

        if buy_prob > 0.55:
            action = "BUY"
            stop_loss = current_price - (1.5 * current_atr)
            explanation = f"Model sees {buy_prob*100:.1f}% upside probability (>{self.THRESHOLD*100}% threshold)"
        elif buy_prob < 0.45:
            action = "SELL"
            stop_loss = current_price + (1.5 * current_atr)
            explanation = f"Model sees {(1-buy_prob)*100:.1f}% downside probability"
        else:
            action = "HOLD"
            stop_loss = None
            explanation = "Probability near 50/50 — no clear edge"

        return {
            "action": action,
            "confidence": round(buy_prob * 100, 2) if action == "BUY" else round((1 - buy_prob) * 100, 2),
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "current_price": round(current_price, 2),
            "buy_prob": float(buy_prob),
            "explanation": explanation,
            "model_used": self.model_name or "Unknown"
        }

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------
    def save_model(self, ticker):
        """Save trained model + metadata to disk."""
        if self.model is None:
            logger.warning("No model to save")
            return False

        path = os.path.join(MODEL_DIR, f'{ticker.upper()}.joblib')
        payload = {
            'model': self.model,
            'model_name': self.model_name,
            'features': self.features,
            'metrics': self.metrics,
            'feature_importances': self.feature_importances,
        }
        joblib.dump(payload, path)
        logger.info(f"Model saved → {path}")
        return True

    def load_model(self, ticker):
        """Load a previously saved model. Returns True if successful."""
        path = os.path.join(MODEL_DIR, f'{ticker.upper()}.joblib')
        if not os.path.exists(path):
            return False

        try:
            payload = joblib.load(path)
            self.model = payload['model']
            self.model_name = payload.get('model_name', 'Unknown')
            self.features = payload.get('features', self.FEATURES)
            self.metrics = payload.get('metrics', {})
            self.feature_importances = payload.get('feature_importances', {})
            logger.info(f"Model loaded ← {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model for {ticker}: {e}")
            return False

    # ------------------------------------------------------------------
    # Convenience: get all evaluation info
    # ------------------------------------------------------------------
    def get_evaluation_summary(self):
        """Returns a dict with all metrics, comparison results, and feature importances."""
        return {
            'best_model': self.model_name,
            'metrics': self.metrics,
            'model_comparison': getattr(self, 'model_comparison', {}),
            'feature_importances': self.feature_importances,
        }


# ==================================================================
# Quick self-test
# ==================================================================
if __name__ == "__main__":
    from data_fetcher import fetch_stock_data, add_technical_indicators

    print("=" * 60)
    print("  Stock Predictor — Model Test")
    print("=" * 60)

    df, cur = fetch_stock_data("TSLA")
    df = add_technical_indicators(df)

    predictor = StockPredictor()
    success_rate = predictor.train_and_evaluate(df)
    print(f"\nHistorical Precision: {success_rate}%")

    summary = predictor.get_evaluation_summary()
    print(f"Best Model: {summary['best_model']}")
    print(f"All Metrics: {summary['metrics']}")
    print(f"Feature Importances (top 5):")
    for feat, imp in list(summary['feature_importances'].items())[:5]:
        print(f"  {feat:20s} → {imp:.4f}")

    prediction = predictor.predict_latest(df)
    print(f"\nLatest Prediction: {prediction}")

    # Test persistence
    predictor.save_model("TSLA")
    new_predictor = StockPredictor()
    loaded = new_predictor.load_model("TSLA")
    print(f"\nModel load test: {'✅ Success' if loaded else '❌ Failed'}")
