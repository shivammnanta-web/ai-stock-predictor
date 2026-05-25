# StockSense AI

StockSense AI is a full stack intelligent stock analysis platform that generates Buy Sell and Hold signals using machine learning technical indicators and real time news sentiment. The system is designed to simulate a real world trading intelligence pipeline.

---

## Overview

The application combines data driven machine learning with contextual news analysis to produce more informed trading signals. It focuses on improving decision quality instead of chasing unrealistic accuracy.

---

## Features

Multi model machine learning engine  
Real time stock data analysis  
Advanced technical indicators  
News sentiment analysis with recency weighting  
Hybrid decision system combining ML and sentiment  
Feedback driven retraining system  
Backtesting engine for performance evaluation  
Model persistence for fast predictions  

---

## Tech Stack

Backend  
Python  
Flask  

Machine Learning  
scikit learn  
XGBoost  
pandas  
numpy  

Natural Language Processing  
nltk  

Data Source  
yfinance  

Deployment  
Vercel  

---

## How It Works

### Data Processing
The system fetches stock data and generates technical indicators such as SMA RSI MACD Bollinger Bands ATR volatility and momentum.

### Machine Learning Engine
Multiple models are trained including Random Forest Logistic Regression and XGBoost. Time series validation is used to evaluate performance and the best model is selected automatically.

### Sentiment Analysis
The system fetches recent news and applies sentiment scoring. More recent news is given higher importance.

### Decision System
Machine learning predictions and sentiment scores are combined to generate a final trading signal.

### Feedback Learning
Users can validate predictions and the system retrains using this feedback to improve performance.

### Backtesting
Trading strategies are simulated on historical data to evaluate real world effectiveness.

---

## Project Structure

StockSense AI

app.py
index.py
model.py
data_fetcher.py
news_analyzer.py
feedback_store.py

services
utils

saved_models
assets
requirements.txt


---

## Installation

https://github.com/shivammnanta-web/ai-stock-predictor


## Install dependencies

pip install -r requirements.txt

## Run the application

python app.py


---

## API Endpoints

api analyze ticker  
api backtest ticker  
api feedback  
api retrain ticker  
api search  

---

## Example Output

{
"action": "BUY",
"confidence": 68.4,
"stop_loss": 245.32
}


---

## Disclaimer

This project is for educational purposes only and does not provide financial advice.

---

## Author

Shivam  
shivammnanta@gmail.com  

---

## Support

If you find this project useful consider giving it a star on GitHub


