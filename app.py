from flask import Flask, jsonify, request
import joblib
import yfinance as yf
import pandas as pd
from features import add_features

app = Flask(__name__)
model = joblib.load("xgb_model.pkl")

@app.route("/")
def home():
    return "<h2>✅ Stock Trend Predictor API is Running!<br>Use /predict?ticker=AAPL</h2>"

@app.route("/predict")
def predict():
    ticker = request.args.get("ticker", "AAPL")
    df = yf.download(ticker, period="6mo")
    df = add_features(df)
    last_row = df.dropna().iloc[-1]

    X = pd.DataFrame([last_row[["Close","Return","SMA5","SMA10","EMA20","Volatility"]]])
    pred = model.predict(X)[0]
    return jsonify({"ticker": ticker, "prediction": "Up" if pred == 1 else "Down"})

if __name__ == "__main__":
    app.run(debug=True)
