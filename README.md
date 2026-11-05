# Stock Trend Predictor

A Streamlit dashboard that forecasts whether a stock will close higher or lower in the next trading session. It also shows how often that forecast has been right on days the model never saw.

**Live app:** https://stocktrend-predict.streamlit.app

It works with any Yahoo Finance symbol: US stocks (`AAPL`, `MSFT`), NSE and BSE stocks (`RELIANCE.NS`, `TCS.BO`), indices (`^NSEI`, `^GSPC`) and ETFs.

## Pages

- **Overview:** the quote with a 3-month sparkline, previous close, day change, 52-week range, volatility, RSI and volume. Below it:
  - a price chart (1M to 10Y) with SMA/EMA overlays and volume coloured by up or down day
  - the next-session forecast and the model's track record on the stock
  - tabs for model testing, indicators and the full price table, with CSV downloads
- **Compare:** up to six symbols side by side, with quick lists (US tech, India IT, India banks, indices):
  - tiles coloured by today's move
  - a growth-of-100 chart
  - a scorecard of return, volatility, drawdown, RSI and 52-week position
  - a correlation grid
- **Strategy lab:** turn the model's calls into a trading rule. Set the starting amount, the probability needed to buy, the cost per trade, and whether to go short on Down calls. You see the final value, a comparison with buy and hold, the trade count, winning days, drawdown, an equity curve with holding periods shaded, and a trade log.
- **Guess the move:** a game. See a real chart up to a random day from the test year, guess Higher or Lower, and compare your score and streak with the model's.
- **Learn:** RSI, moving averages, MACD, Bollinger bands and walk-forward testing, explained on a live chart of the chosen symbol, plus a four-question quiz.

The symbol you pick is shared across pages, and every number comes from the same cached data.

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
| `dashboard.py` | Streamlit entry point and page navigation |
| `views/` | One file per page: overview, compare, strategy, game, learn |
| `ui.py` | Shared styling, data loading, charts and page blocks |
| `features.py` | Yahoo Finance loading, data cleaning, indicators and model features |
| `train_model.py` | Model, walk-forward test, metrics, command-line report |
| `app.py` | Flask API |
| `fetch_data.py` | Quick check that data loads for a symbol |
| `.streamlit/config.toml` | Theme (colours and IBM Plex fonts) |

## Tech stack

Python, Streamlit (charts drawn with Altair, which ships with Streamlit), yfinance, pandas, XGBoost, scikit-learn and Flask.
