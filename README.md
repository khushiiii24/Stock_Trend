# Stock Trend Predictor

A Streamlit dashboard that forecasts whether a stock will close higher or lower in the next trading session. It also shows how often that forecast has been right on days the model never saw.

It works with any Yahoo Finance symbol: US stocks (`AAPL`, `MSFT`), NSE and BSE stocks (`RELIANCE.NS`, `TCS.BO`), indices (`^NSEI`, `^GSPC`) and ETFs.

## What the dashboard shows

- **Quote summary:** last price and daily change, previous close, day range, 52-week range with position, 20-day annualised volatility, RSI (14) and average volume. If the market is still open, the latest price is marked as live.
- **Price chart:** 1M to 10Y range with optional 5-day SMA, 10-day SMA and 20-day EMA overlays, a volume panel and a hover readout of OHLC values.
- **Next-session call:** Up or Down, the probability of a higher close, how far that probability is from 50%, and the model's track record on this stock.
- **Model track record:** walk-forward accuracy with a 95% confidence interval, comparison with a naive baseline, ROC AUC, hit rate on high-conviction calls, a calls-vs-outcomes table, accuracy per test block, a growth-of-100 backtest against buy and hold, and feature importance.
- **Indicators:** RSI chart and today's value of every model input, with its usual range and percentile in the stock's own history.
- **Price data:** daily table with indicators, plus CSV downloads of the full history and of every test prediction.

## How the model works

**Inputs.** There are 12 features. All of them are scale-free, so a $20 stock and a ₹3,000 stock mean the same thing to the model:

- 1-day and 5-day returns
- Close relative to its 5-day SMA, 10-day SMA and 20-day EMA
- 5-day vs 10-day SMA
- 10-day volatility
- RSI (14)
- MACD histogram
- Position in the Bollinger band
- Average true range
- Volume relative to its 20-day average

**Target.** The target is 1 if the next session's dividend-adjusted close is higher than today's, otherwise 0. The most recent session has no label yet, so it is used only for the live forecast.

**Model.** A small, heavily regularised XGBoost classifier, retrained for each symbol on up to 10 years of daily data.

**Testing.** Accuracy comes from a walk-forward test, so every scored prediction is for a day the model never saw:

1. Take the last 250 sessions (about one year).
2. Split them into five consecutive blocks.
3. For each block, train only on the sessions before it, then predict the block.

**Baseline.** The model is compared with the simplest strategy: predicting whichever direction was more common in the test period (usually "always Up").

**Prices.** Displayed prices are as quoted (split-adjusted), so they match Yahoo Finance and broker apps. Returns and model inputs use dividend-adjusted prices, so an ex-dividend drop is not mistaken for a price fall.

**What to expect.** Daily direction is close to random. Accuracy in the low-to-mid 50s is normal for this kind of model, and on many stocks the edge over the baseline falls inside the margin of error. The dashboard says when that happens. This is an educational project, not investment advice.

## Run it

```bash
git clone https://github.com/khushiiii24/Stock_Trend.git
cd Stock_Trend

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
streamlit run dashboard.py
```

Data is cached for 15 minutes per symbol. Loading a new symbol takes a few seconds while the model is tested and trained.

### Command-line report

```bash
python train_model.py AAPL
python train_model.py RELIANCE.NS --period 5y
```

This prints the full evaluation and saves the fitted model to `xgb_model.pkl`.

### JSON API

```bash
python app.py
# http://127.0.0.1:5000/predict?ticker=TCS.NS
```

The response contains the forecast, `prob_up`, the latest close and the walk-forward track record.

## Files

| File | Purpose |
|---|---|
| `dashboard.py` | Streamlit app |
| `features.py` | Yahoo Finance loading, data cleaning, indicators and model features |
| `train_model.py` | Model, walk-forward test, metrics, command-line report |
| `app.py` | Flask API |
| `fetch_data.py` | Quick check that data loads for a symbol |
| `.streamlit/config.toml` | Theme (colours and IBM Plex fonts) |

## Tech stack

Python, Streamlit (charts drawn with Altair, which ships with Streamlit), yfinance, pandas, XGBoost, scikit-learn and Flask.
