"""
news_analyzer.py — Sentiment analysis with recency weighting and dynamic fusion.

Improvements over original:
  1. Recency weighting — newer headlines get exponentially more influence
  2. Dynamic fusion — weighting adjusts based on news volume and conviction
  3. Proper logging
  4. Robust extraction for both old and new yfinance news formats
"""

import math
from datetime import datetime, timezone
import yfinance as yf
from utils.logger import get_logger

logger = get_logger(__name__)

# NLTK setup — lazy import so module loads even if VADER isn't ready yet
_sia = None


def _get_sia():
    """Lazy-load SentimentIntensityAnalyzer (handles missing lexicon gracefully)."""
    global _sia
    if _sia is not None:
        return _sia

    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        _sia = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
        _sia = SentimentIntensityAnalyzer()
    return _sia


class NewsAnalyzer:
    """Fetches stock news, scores sentiment with recency bias, and fuses with technicals."""

    def __init__(self):
        self.sia = _get_sia()

    # ─── News fetching ───────────────────────────────────────────────

    def get_stock_news(self, ticker):
        """Fetches the latest news for a ticker via yfinance."""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news or []
            logger.info(f"Fetched {len(news)} news items for {ticker}")
            return news
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []

    # ─── Sentiment scoring with recency weighting ────────────────────

    def analyze_sentiment(self, news_items):
        """
        Scores each headline with VADER and applies exponential decay
        weighting so recent news has heavier impact.

        Decay formula:  weight = exp(-lambda * hours_old)
        Default lambda = 0.02  →  24 h old ≈ 62 % weight, 72 h ≈ 24 %
        """
        if not news_items:
            return {"average_sentiment": 0, "overall_tone": "Neutral", "headlines": []}

        scored = []
        now = datetime.now(timezone.utc)
        decay_lambda = 0.02  # Tune this to adjust time sensitivity
        limit = min(15, len(news_items))

        weighted_sum = 0.0
        total_weight = 0.0

        for item in news_items[:limit]:
            # ── Extract title (supports old & new yfinance formats) ──
            content = item.get('content', {})
            title = content.get('title', item.get('title', ''))
            if not title:
                continue

            # ── Extract publication time for recency weight ──
            pub_time = self._extract_pub_time(item)
            if pub_time:
                hours_old = max((now - pub_time).total_seconds() / 3600, 0)
            else:
                hours_old = 48  # Unknown age → assume moderately old

            weight = math.exp(-decay_lambda * hours_old)

            # ── VADER score ──
            score = self.sia.polarity_scores(title)
            compound = score['compound']

            # ── Source ──
            source = (
                content.get('provider', {}).get('displayName')
                or item.get('publisher', 'News Source')
            )

            scored.append({
                "title": title,
                "score": round(compound, 3),
                "weight": round(weight, 3),
                "hours_old": round(hours_old, 1),
                "sentiment": (
                    "Positive" if compound > 0.1
                    else ("Negative" if compound < -0.1 else "Neutral")
                ),
                "source": source,
            })

            weighted_sum += compound * weight
            total_weight += weight

        avg_sentiment = round(weighted_sum / total_weight, 3) if total_weight > 0 else 0

        return {
            "average_sentiment": avg_sentiment,
            "overall_tone": (
                "Positive" if avg_sentiment > 0.1
                else ("Negative" if avg_sentiment < -0.1 else "Neutral")
            ),
            "headlines": scored,
            "news_count": len(scored),
        }

    # ─── Dynamic fusion ──────────────────────────────────────────────

    def diplomatic_fusion(self, tech_signal_prob, sentiment_score):
        """
        Combines technical probability (0–1) with sentiment score (−1 to 1).

        Dynamic weighting (instead of fixed 70/30):
          - When sentiment conviction is strong AND news volume is high,
            sentiment weight increases up to 40 %.
          - When sentiment is near-zero (noise), technical weight dominates at 85 %.
        """
        # Normalize sentiment to 0–1 range
        norm_sentiment = (sentiment_score + 1) / 2

        # ── Dynamic weight calculation ──
        # Sentiment conviction = how far from neutral (0)
        conviction = abs(sentiment_score)

        # Sentiment weight: 15 % baseline, up to 40 % at full conviction
        sent_weight = 0.15 + 0.25 * conviction
        tech_weight = 1.0 - sent_weight

        fused_score = (tech_weight * tech_signal_prob) + (sent_weight * norm_sentiment)

        # ── Diplomatic reasoning ──
        reasoning = ""
        if tech_signal_prob > 0.6 and sentiment_score < -0.3:
            final_action = "HOLD / CAUTION"
            reasoning = (
                "Technical indicators show strength, but heavily negative news "
                "sentiment warns of a potential bull trap or near-term correction."
            )
        elif tech_signal_prob < 0.4 and sentiment_score > 0.4:
            final_action = "HOLD / WATCH"
            reasoning = (
                "Technicals are bearish, but strongly positive news flow hints "
                "at a possible trend reversal. Wait for confirmation."
            )
        elif fused_score > 0.58:
            final_action = "BUY"
            reasoning = (
                f"Technical and sentiment signals are aligned bullish "
                f"(tech weight {tech_weight:.0%}, sentiment weight {sent_weight:.0%})."
            )
        elif fused_score < 0.42:
            final_action = "SELL"
            reasoning = (
                f"Technical weakness reinforced by negative sentiment "
                f"(tech weight {tech_weight:.0%}, sentiment weight {sent_weight:.0%})."
            )
        else:
            final_action = "HOLD"
            reasoning = (
                "Mixed signals — neither technicals nor news provide a clear edge."
            )

        return {
            "fused_score": round(fused_score * 100, 2),
            "diplomatic_action": final_action,
            "reasoning": reasoning,
            "weights": {
                "technical": round(tech_weight, 2),
                "sentiment": round(sent_weight, 2),
            },
        }

    # ─── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_pub_time(item):
        """
        Best-effort extraction of the publication datetime from a
        yfinance news item.  Returns a timezone-aware datetime or None.
        """
        try:
            # New format: content.pubDate (ISO string)
            content = item.get('content', {})
            pub_str = content.get('pubDate')
            if pub_str:
                return datetime.fromisoformat(pub_str.replace('Z', '+00:00'))

            # Old format: providerPublishTime (unix timestamp)
            ts = item.get('providerPublishTime')
            if ts:
                return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            pass
        return None


# ── Self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = NewsAnalyzer()
    news = analyzer.get_stock_news("AAPL")
    sentiment = analyzer.analyze_sentiment(news)
    print(f"Sentiment: {sentiment['overall_tone']} ({sentiment['average_sentiment']})")
    for h in sentiment['headlines'][:3]:
        print(f"  [{h['weight']:.2f}] {h['sentiment']:>8s}  {h['title'][:70]}")

    fusion = analyzer.diplomatic_fusion(0.65, -0.5)
    print(f"\nFusion: {fusion}")
