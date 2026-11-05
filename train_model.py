"""Train and evaluate the next-session direction model.

The dashboard and the API import ``analyze``. Run this file directly to print an
evaluation report for one ticker and save the fitted model:

    python train_model.py AAPL
    python train_model.py RELIANCE.NS --period 5y --out xgb_model.pkl

How accuracy is measured (walk-forward test): the last ``TEST_DAYS`` sessions are
split into ``FOLDS`` consecutive blocks. For each block the model is retrained on
every session *before* it and then predicts the block, so every scored prediction
is for a day the model had never seen. The final model used for the live call is
then fitted on all labelled history.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from features import FEATURE_LABELS, FEATURES, add_features, load_history, session_in_progress

# Deliberately small and heavily regularised: daily returns are mostly noise, and a
# bigger model just memorises it and becomes over-confident.
MODEL_PARAMS = dict(
    n_estimators=200,
    max_depth=2,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=30,
    reg_lambda=10.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
)

TEST_DAYS = 250          # about one trading year
FOLDS = 5                # retrain five times across the test year
MIN_TRAIN_ROWS = 250     # never fit on less than a year of sessions
CONVICTION = 0.05        # |P(up) - 50%| >= 5 points counts as a high-conviction call


def build_model() -> XGBClassifier:
    return XGBClassifier(**MODEL_PARAMS)


class _ConstantModel:
    """Fallback when a training window holds only one class (rare, very short histories)."""

    def __init__(self, label: int):
        self.label = int(label)

    def predict_proba(self, X):
        p = np.full(len(X), float(self.label))
        return np.column_stack([1 - p, p])


def _fit(train: pd.DataFrame):
    y = train["Target"].astype(int)
    if y.nunique() < 2:
        return _ConstantModel(y.iloc[0])
    return build_model().fit(train[FEATURES], y)


def _labeled(data: pd.DataFrame) -> pd.DataFrame:
    return data[data["Target"].notna()]


def is_up_call(prob_up) -> np.ndarray | bool:
    return np.asarray(prob_up) >= 0.5


def walk_forward(data: pd.DataFrame, test_days: int = TEST_DAYS, folds: int = FOLDS) -> pd.DataFrame:
    """Out-of-sample predictions for the most recent ``test_days`` labelled sessions."""
    labeled = _labeled(data)
    n = len(labeled)
    test_days = min(test_days, n - MIN_TRAIN_ROWS)
    if test_days < 40:
        raise ValueError(
            f"Need at least {MIN_TRAIN_ROWS + 40} sessions of history to test the model; "
            f"this symbol has {n}."
        )

    edges = np.linspace(n - test_days, n, folds + 1).round().astype(int)
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        train, test = labeled.iloc[:a], labeled.iloc[a:b]
        prob = _fit(train).predict_proba(test[FEATURES])[:, 1]
        parts.append(
            pd.DataFrame(
                {
                    "ProbUp": prob,
                    "Actual": test["Target"].astype(int).to_numpy(),
                    "NextReturn": test["NextReturn"].to_numpy(),
                    "TrainRows": a,
                },
                index=test.index,
            )
        )

    bt = pd.concat(parts)
    bt["Predicted"] = is_up_call(bt["ProbUp"]).astype(int)
    bt["Correct"] = (bt["Predicted"] == bt["Actual"]).astype(int)
    bt["ModelReturn"] = np.where(bt["Predicted"] == 1, bt["NextReturn"], 0.0)
    bt["ModelEquity"] = 100 * (1 + bt["ModelReturn"]).cumprod()
    bt["BuyHoldEquity"] = 100 * (1 + bt["NextReturn"]).cumprod()
    return bt


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a hit rate (Wilson score)."""
    if n == 0:
        return (math.nan, math.nan)
    p = hits / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def summarize(bt: pd.DataFrame) -> dict:
    y = bt["Actual"].to_numpy()
    pred = bt["Predicted"].to_numpy()
    prob = bt["ProbUp"].to_numpy()
    n = len(bt)
    hits = int(bt["Correct"].sum())

    up_share = float(y.mean())
    conviction = np.abs(prob - 0.5) >= CONVICTION
    up_calls, down_calls = pred == 1, pred == 0
    ci_low, ci_high = wilson_interval(hits, n)

    return {
        "test_days": n,
        "test_start": bt.index[0],
        "test_end": bt.index[-1],
        "accuracy": hits / n,
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
        "up_share": up_share,
        "baseline": max(up_share, 1 - up_share),
        "baseline_label": "always Up" if up_share >= 0.5 else "always Down",
        "precision_up": float(y[up_calls].mean()) if up_calls.any() else math.nan,
        "precision_down": float(1 - y[down_calls].mean()) if down_calls.any() else math.nan,
        "up_calls": int(up_calls.sum()),
        "down_calls": int(down_calls.sum()),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else math.nan,
        "conviction_n": int(conviction.sum()),
        "conviction_accuracy": float(bt["Correct"][conviction].mean()) if conviction.any() else math.nan,
        "true_up": int(((pred == 1) & (y == 1)).sum()),
        "false_up": int(((pred == 1) & (y == 0)).sum()),
        "true_down": int(((pred == 0) & (y == 0)).sum()),
        "false_down": int(((pred == 0) & (y == 1)).sum()),
        "model_return": float(bt["ModelEquity"].iloc[-1] / 100 - 1),
        "buyhold_return": float(bt["BuyHoldEquity"].iloc[-1] / 100 - 1),
        "model_drawdown": max_drawdown(bt["ModelEquity"]),
        "buyhold_drawdown": max_drawdown(bt["BuyHoldEquity"]),
        "exposure": float(pred.mean()),
    }


def feature_importance(model) -> pd.Series:
    """Share of total split gain per feature (sums to 1)."""
    if not hasattr(model, "get_booster"):
        return pd.Series(0.0, index=FEATURES)
    gain = model.get_booster().get_score(importance_type="gain")
    s = pd.Series({f: gain.get(f, 0.0) for f in FEATURES}, dtype=float)
    total = s.sum()
    return (s / total if total > 0 else s).sort_values(ascending=False)


@dataclass
class Analysis:
    prob_up: float
    as_of: pd.Timestamp
    backtest: pd.DataFrame
    metrics: dict
    importance: pd.Series
    train_rows: int
    train_start: pd.Timestamp
    model: object

    @property
    def call(self) -> str:
        return "Up" if is_up_call(self.prob_up) else "Down"


def analyze(data: pd.DataFrame, live_last_bar: bool = False) -> Analysis:
    """Walk-forward test, then fit on all history and score the latest session.

    ``live_last_bar``: the last row is a session still in progress. Its close is a
    live price, so the label of the row before it is not final yet and is dropped.
    """
    data = data.copy()
    if live_last_bar and len(data) >= 2:
        data.loc[data.index[-2], ["Target", "NextReturn"]] = np.nan

    bt = walk_forward(data)
    labeled = _labeled(data)
    model = _fit(labeled)
    prob = float(model.predict_proba(data[FEATURES].iloc[[-1]])[0, 1])

    return Analysis(
        prob_up=prob,
        as_of=data.index[-1],
        backtest=bt,
        metrics=summarize(bt),
        importance=feature_importance(model),
        train_rows=len(labeled),
        train_start=labeled.index[0],
        model=model,
    )


def _pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{digits}f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the next-session trend model.")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Yahoo Finance symbol, e.g. AAPL or TCS.NS")
    parser.add_argument("--period", default="10y", help="History to download (default 10y)")
    parser.add_argument("--out", default="xgb_model.pkl", help="Where to save the fitted model")
    args = parser.parse_args()

    prices, meta = load_history(args.ticker, args.period)
    data = add_features(prices)
    res = analyze(data, live_last_bar=session_in_progress(prices, meta))
    m = res.metrics

    print(f"\n{meta['name']} ({meta['symbol']}), {meta['exchange']} {meta['currency']}")
    print(f"Trained on {res.train_rows:,} sessions, {res.train_start:%d %b %Y} to {res.as_of:%d %b %Y}")
    print(f"\nWalk-forward test, {m['test_days']} sessions ({m['test_start']:%d %b %Y} to {m['test_end']:%d %b %Y})")
    print(f"  Accuracy            {_pct(m['accuracy'])}  (95% CI {_pct(m['accuracy_ci_low'])} to {_pct(m['accuracy_ci_high'])})")
    print(f"  Baseline ({m['baseline_label']}) {_pct(m['baseline'])}")
    print(f"  Up calls right      {_pct(m['precision_up'])} of {m['up_calls']}")
    print(f"  Down calls right    {_pct(m['precision_down'])} of {m['down_calls']}")
    print(f"  ROC AUC             {m['auc']:.3f}")
    print(f"  High-conviction     {_pct(m['conviction_accuracy'])} of {m['conviction_n']} calls")
    print(f"  Model signal return {_pct(m['model_return'])} (max drawdown {_pct(m['model_drawdown'])})")
    print(f"  Buy and hold return {_pct(m['buyhold_return'])} (max drawdown {_pct(m['buyhold_drawdown'])})")
    print(f"\nNext session after {res.as_of:%d %b %Y}: {res.call}, P(up) = {_pct(res.prob_up)}")
    print("\nTop inputs by gain:")
    for name, share in res.importance.head(5).items():
        print(f"  {FEATURE_LABELS[name]:<28} {_pct(share)}")

    joblib.dump(
        {
            "model": res.model,
            "features": FEATURES,
            "ticker": meta["symbol"],
            "trained_through": res.as_of,
            "metrics": m,
            "params": MODEL_PARAMS,
        },
        args.out,
    )
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
