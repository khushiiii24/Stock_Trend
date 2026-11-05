"""Price loading and feature engineering shared by the dashboard, the API and the trainer.

Two price series are kept on purpose:

* ``Close`` / ``High`` / ``Low`` are the prices exchanges and quote sites show
  (split-adjusted, not dividend-adjusted). Everything displayed uses these, so
  the numbers match what you see on Yahoo Finance or your broker.
* ``Adj Close`` also folds in dividends. Returns, the model's inputs and the
  prediction target use it, so an ex-dividend drop is not mistaken for a
  falling price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# Model inputs. Every one is a ratio or a bounded oscillator rather than a raw
# price, so a model trained on a $20 stock means the same thing on a $2,000 one.
FEATURES = [
    "Return1D",
    "Return5D",
    "CloseVsSMA5",
    "CloseVsSMA10",
    "CloseVsEMA20",
    "SMA5VsSMA10",
    "Volatility10D",
    "RSI14Adj",
    "MACDHist",
    "BollingerPos",
    "ATRPct",
    "VolumeRatio",
]

FEATURE_LABELS = {
    "Return1D": "1-day return",
    "Return5D": "5-day return",
    "CloseVsSMA5": "Close vs 5-day average",
    "CloseVsSMA10": "Close vs 10-day average",
    "CloseVsEMA20": "Close vs 20-day EMA",
    "SMA5VsSMA10": "5-day vs 10-day average",
    "Volatility10D": "10-day volatility",
    "RSI14Adj": "RSI (14)",
    "MACDHist": "MACD histogram",
    "BollingerPos": "Position in Bollinger band",
    "ATRPct": "Average true range",
    "VolumeRatio": "Volume vs 20-day average",
}

# Rows needed before every indicator is defined (MACD 26 + signal 9, rounded up).
WARMUP_ROWS = 40


class DataError(Exception):
    """Raised when price data can't be fetched or is unusable."""


def load_history(ticker: str, period: str = "10y") -> tuple[pd.DataFrame, dict]:
    """Download daily prices for ``ticker`` and return (clean OHLCV frame, metadata)."""
    symbol = ticker.strip().upper()
    if not symbol:
        raise DataError("Enter a ticker symbol.")

    tk = yf.Ticker(symbol)
    try:
        raw = tk.history(period=period, interval="1d", auto_adjust=False, actions=False)
    except Exception as exc:  # yfinance raises a handful of different errors
        raise DataError(f"Yahoo Finance did not return prices for {symbol}: {exc}") from exc

    if raw is None or raw.empty:
        raise DataError(
            f"No price history found for {symbol}. Check the symbol; Indian stocks "
            "need an exchange suffix such as RELIANCE.NS or TCS.BO."
        )

    try:
        meta = tk.history_metadata or {}
    except Exception:
        meta = {}

    return clean_ohlcv(raw), summarize_metadata(symbol, meta)


def clean_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise yfinance output (``Ticker.history`` or ``yf.download``) to one tidy frame."""
    df = raw.copy()

    # yf.download returns (field, ticker) MultiIndex columns in recent versions.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]

    if "Close" not in df.columns:
        raise DataError("Price data has no Close column.")
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    for col in ("Open", "High", "Low"):
        if col not in df.columns:
            df[col] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    # Exchange-local calendar dates, no timezone, one row per session.
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    df = df[df["Close"].notna() & (df["Close"] > 0)]
    for col in ("Open", "High", "Low"):
        df[col] = df[col].fillna(df["Close"])
    df["Adj Close"] = df["Adj Close"].fillna(df["Close"])
    df["Volume"] = df["Volume"].fillna(0.0)
    return df


def summarize_metadata(symbol: str, meta: dict) -> dict:
    """Keep the few metadata fields the UI needs, as plain picklable values."""

    def ts(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return pd.Timestamp(int(value), unit="s", tz="UTC")
        stamp = pd.Timestamp(value)
        return stamp if stamp.tzinfo else stamp.tz_localize("UTC")

    regular = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    return {
        "symbol": symbol,
        "name": meta.get("longName") or meta.get("shortName") or symbol,
        "currency": meta.get("currency") or "",
        "exchange": meta.get("fullExchangeName") or meta.get("exchangeName") or "",
        "instrument": meta.get("instrumentType") or "",
        "timezone": meta.get("exchangeTimezoneName") or "UTC",
        "session_start": ts(regular.get("start")),
        "session_end": ts(regular.get("end")),
        "last_trade_time": ts(meta.get("regularMarketTime")),
    }


def session_in_progress(df: pd.DataFrame, meta: dict, now: pd.Timestamp | None = None) -> bool:
    """True when the last row is today's still-open session (its Close is a live price)."""
    start, end = meta.get("session_start"), meta.get("session_end")
    if start is None or end is None or df.empty:
        return False
    now = now or pd.Timestamp.now(tz="UTC")
    local_day = start.tz_convert(meta.get("timezone") or "UTC").tz_localize(None).normalize()
    return bool(start <= now < end and df.index[-1] == local_day)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, the version charting platforms show."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Chart indicators on quoted prices (they match charting sites). Keeps every row."""
    out = clean_ohlcv(df)
    close = out["Close"]
    out["Change"] = close.diff()
    out["ChangePct"] = close.pct_change()
    out["SMA5"] = close.rolling(5).mean()
    out["SMA10"] = close.rolling(10).mean()
    out["EMA20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    out["RSI14"] = rsi(close, 14)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add chart indicators, model features and the next-day target.

    Warm-up rows (where an indicator is not defined yet) are dropped. The most
    recent row keeps ``Target = NaN`` (tomorrow is unknown) so it is never used
    for training, only for the live prediction.
    """
    out = add_indicators(df)
    close = out["Close"]

    # Model features on dividend-adjusted prices.
    factor = out["Adj Close"] / close
    adj = out["Adj Close"]
    high, low = out["High"] * factor, out["Low"] * factor
    ret = adj.pct_change()
    sma5, sma10 = adj.rolling(5).mean(), adj.rolling(10).mean()
    ema20 = adj.ewm(span=20, adjust=False).mean()

    out["Return"] = ret
    out["Return1D"] = ret
    out["Return5D"] = adj.pct_change(5)
    out["CloseVsSMA5"] = adj / sma5 - 1
    out["CloseVsSMA10"] = adj / sma10 - 1
    out["CloseVsEMA20"] = adj / ema20 - 1
    out["SMA5VsSMA10"] = sma5 / sma10 - 1
    out["Volatility10D"] = ret.rolling(10).std()
    out["Volatility"] = out["Volatility10D"]
    out["RSI14Adj"] = rsi(adj, 14)

    macd = adj.ewm(span=12, adjust=False).mean() - adj.ewm(span=26, adjust=False).mean()
    out["MACDHist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / adj

    mid, sd = adj.rolling(20).mean(), adj.rolling(20).std()
    out["BollingerPos"] = (adj - (mid - 2 * sd)) / (4 * sd)

    prev = adj.shift(1)
    true_range = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    out["ATRPct"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / adj

    # Indices and FX often report zero volume; treat that as "normal volume".
    vol = out["Volume"].where(out["Volume"] > 0)
    out["VolumeRatio"] = (vol / vol.rolling(20, min_periods=5).mean()).fillna(1.0)

    # Target: did the next session close higher (dividend-adjusted)?
    out["NextReturn"] = adj.shift(-1) / adj - 1
    out["Target"] = np.where(out["NextReturn"].notna(), (out["NextReturn"] > 0).astype(float), np.nan)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.iloc[WARMUP_ROWS:] if len(out) > WARMUP_ROWS else out.iloc[0:0]
    return out.dropna(subset=FEATURES)
