"""Quick smoke test for the upgraded ML pipeline."""
import sys
sys.path.insert(0, '.')

from model import StockPredictor
from data_fetcher import fetch_stock_data, add_technical_indicators

print("=" * 60)
print("  Smoke Test: ML Pipeline")
print("=" * 60)

# 1. Fetch data
print("\n[1] Fetching AAPL data...")
df, currency = fetch_stock_data("AAPL")
if df is None:
    print("FAIL: Could not fetch data")
    sys.exit(1)
print(f"    Fetched {len(df)} rows (currency: {currency})")

# 2. Add indicators
print("\n[2] Adding technical indicators...")
df = add_technical_indicators(df)
print(f"    After indicators: {len(df)} rows")
print(f"    Columns: {list(df.columns)}")

# 3. Train model
print("\n[3] Training model with TimeSeriesSplit CV...")
predictor = StockPredictor()
rate = predictor.train_and_evaluate(df)
print(f"    Precision: {rate}%")

# 4. Get evaluation summary
summary = predictor.get_evaluation_summary()
print(f"\n[4] Best model: {summary['best_model']}")
print(f"    Metrics: {summary['metrics']}")
print(f"    Model comparison:")
for name, metrics in summary.get('model_comparison', {}).items():
    print(f"      {name:20s} -> {metrics}")
print(f"    Top features:")
for feat, imp in list(summary['feature_importances'].items())[:5]:
    print(f"      {feat:20s} -> {imp:.4f}")

# 5. Predict
print("\n[5] Latest prediction...")
pred = predictor.predict_latest(df)
for k, v in pred.items():
    print(f"    {k:20s}: {v}")

# 6. Model persistence
print("\n[6] Saving model...")
saved = predictor.save_model("AAPL")
print(f"    Save: {'OK' if saved else 'FAILED'}")

print("\n[7] Loading model...")
new_predictor = StockPredictor()
loaded = new_predictor.load_model("AAPL")
print(f"    Load: {'OK' if loaded else 'FAILED'}")

if loaded:
    pred2 = new_predictor.predict_latest(df)
    print(f"    Loaded model prediction: {pred2['action']} ({pred2['confidence']}%)")

print("\n" + "=" * 60)
print("  All tests passed!" if loaded else "  Some tests failed!")
print("=" * 60)
