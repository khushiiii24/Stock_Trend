"""Small JSON API for the same model the dashboard uses.

Run with:  python app.py
Then open: http://127.0.0.1:5000/predict?ticker=AAPL
"""
from __future__ import annotations

import math
import time

from flask import Flask, jsonify, request

from features import DataError, add_features, load_history, session_in_progress
from train_model import analyze

app = Flask(__name__)

CACHE_SECONDS = 900
_cache: dict[str, tuple[float, dict]] = {}


def _clean(value):
    """JSON-safe numbers: round floats, turn NaN into null."""
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 4)
    return value


def build_report(ticker: str) -> dict:
    prices, meta = load_history(ticker, "10y")
    live = session_in_progress(prices, meta)
    res = analyze(add_features(prices), live_last_bar=live)
    m = res.metrics
    last, prev = float(prices["Close"].iloc[-1]), float(prices["Close"].iloc[-2])
    return {
        "ticker": meta["symbol"],
        "name": meta["name"],
        "exchange": meta["exchange"],
        "currency": meta["currency"],
        "as_of": res.as_of.strftime("%Y-%m-%d"),
        "session_open": live,
        "last_close": round(last, 4),
        "change_pct": round(last / prev - 1, 6),
        "prediction": res.call,
        "prob_up": round(res.prob_up, 4),
        "track_record": {
            "test_sessions": m["test_days"],
            "test_start": m["test_start"].strftime("%Y-%m-%d"),
            "test_end": m["test_end"].strftime("%Y-%m-%d"),
            **{k: _clean(m[k]) for k in (
                "accuracy", "accuracy_ci_low", "accuracy_ci_high", "baseline", "precision_up",
                "precision_down", "auc", "conviction_accuracy", "conviction_n",
                "model_return", "buyhold_return",
            )},
        },
    }


@app.route("/")
def home():
    return (
        "<h2>Stock Trend Predictor API</h2>"
        "<p>Try <a href='/predict?ticker=AAPL'>/predict?ticker=AAPL</a> "
        "or <a href='/predict?ticker=TCS.NS'>/predict?ticker=TCS.NS</a></p>"
    )


@app.route("/predict")
def predict():
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    hit = _cache.get(ticker)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return jsonify(hit[1])
    try:
        report = build_report(ticker)
    except DataError as exc:
        return jsonify({"ticker": ticker, "error": str(exc)}), 404
    except ValueError as exc:  # not enough history to test the model
        return jsonify({"ticker": ticker, "error": str(exc)}), 422
    _cache[ticker] = (time.time(), report)
    return jsonify(report)


if __name__ == "__main__":
    app.run(debug=True)
