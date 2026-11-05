"""Shared look, data loading and chart builders for every page of the app."""
from __future__ import annotations

import base64
import html
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from features import (FEATURE_LABELS, FEATURES, DataError, add_features, add_indicators, load_history,
                      session_in_progress)
from train_model import CONVICTION, analyze

# ---------------------------------------------------------------- tokens
NAVY = "#14213D"
COBALT = "#3355D6"
UP, DOWN = "#12804F", "#C23A2F"
UP_BG, DOWN_BG = "#E8F6EE", "#FCECEA"
AMBER, AMBER_BG = "#9A6200", "#FDF3DF"
SERIES = {"Close": "#2a78d6", "SMA5": "#eb6834", "SMA10": "#1baf7a", "EMA20": "#eda100"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed order, never cycled
OVERLAY_LABELS = {"SMA5": "5-day SMA", "SMA10": "10-day SMA", "EMA20": "20-day EMA"}
BENCHMARK = "#8E98A6"
RULE = "#8E98A6"
INK_RULE = "#5B6676"
Y_EXTENT = 58  # fixed y-axis width so stacked charts line up

RANGES = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "2Y": pd.DateOffset(years=2),
    "5Y": pd.DateOffset(years=5),
    "10Y": pd.DateOffset(years=10),
}
POPULAR = ["AAPL", "MSFT", "NVDA", "TSLA", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "^NSEI"]
CURRENCY_SYMBOLS = {
    "USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥",
    "HKD": "HK$", "CAD": "C$", "AUD": "A$", "SGD": "S$", "CHF": "CHF ", "KRW": "₩",
}
PCT_FEATURES = {
    "Return1D", "Return5D", "CloseVsSMA5", "CloseVsSMA10", "CloseVsEMA20",
    "SMA5VsSMA10", "Volatility10D", "ATRPct", "MACDHist",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
[data-testid="stMainBlockContainer"], .block-container {padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1320px;}
.tp {font-family: "IBM Plex Sans", sans-serif; color: #14213D;}
.tp-up {color: #12804F;} .tp-down {color: #C23A2F;}
.tp-muted {opacity: .66;}

/* page title */
.tp-page {margin: 0 0 14px;}
.tp-page h1 {font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 600; font-size: 2.1rem; line-height: 1.05; margin: 0; padding: 0;}
.tp-page p {font-size: .95rem; opacity: .72; margin: 6px 0 0; max-width: 70ch;}

/* hero band */
.tp-hero {background: #14213D; color: #FFFFFF; border-radius: 14px; padding: 22px 26px 20px;
  display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr) auto; gap: 24px; align-items: end; margin: 4px 0 14px;}
.tp-hero .sym {font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 600; font-size: 3.4rem; line-height: .92; letter-spacing: -.015em;}
.tp-hero .name {font-size: 1.05rem; font-weight: 500; margin-top: 10px;}
.tp-hero .sub {font-size: .85rem; color: #B9C3D6; margin-top: 3px;}
.tp-hero .spark {align-self: center; opacity: .95;}
.tp-hero .spark img {width: 100%; height: 64px; display: block;}
.tp-hero .quote {text-align: right;}
.tp-hero .px {font-size: 2.5rem; font-weight: 500; line-height: 1; letter-spacing: -.01em;}
.tp-chip {display: inline-block; font-size: .86rem; font-weight: 500; padding: 4px 10px; border-radius: 999px; margin-top: 9px; white-space: nowrap;}
.tp-chip.up {background: rgba(91,227,154,.16); color: #6BE7A5;}
.tp-chip.down {background: rgba(255,122,107,.16); color: #FF9487;}
.tp-chip.flat {background: rgba(255,255,255,.12); color: #E6EAF2;}
.tp-chip.live {background: rgba(255,255,255,.12); color: #E6EAF2; font-weight: 400; font-size: .78rem;}
.tp-hero .chips {display: flex; flex-direction: column; align-items: flex-end; gap: 0;}

/* stat tiles */
.tp-tiles {display: grid; grid-template-columns: repeat(var(--cols, 6), minmax(0, 1fr)); gap: 10px; margin: 0 0 20px;}
.tp-tile {background: #FFFFFF; border: 1px solid #DCE2EA; border-radius: 10px; padding: 12px 14px; min-width: 0;}
.tp-tile.up {background: #E8F6EE; border-color: #C7E7D4;}
.tp-tile.down {background: #FCECEA; border-color: #F2CFCA;}
.tp-tile.amber {background: #FDF3DF; border-color: #F1DDB0;}
.tp-tile.cobalt {background: #ECF0FD; border-color: #CDD7F8;}
.tp-tile .label {font-size: .78rem; opacity: .7; margin-bottom: 4px;}
.tp-tile .val {font-size: 1.18rem; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.tp-tile .note {font-size: .78rem; opacity: .7; margin-top: 3px;}
.tp-tile.up .val {color: #0E6B42;} .tp-tile.down .val {color: #A8322A;}
.tp-range {position: relative; height: 5px; border-radius: 3px; margin: 9px 0 6px;
  background: linear-gradient(90deg, #E9A39B, #E7D9A6 50%, #9FD4B5);}
.tp-range i {position: absolute; top: 50%; width: 12px; height: 12px; border-radius: 50%; background: #14213D;
  transform: translate(-50%, -50%); box-shadow: 0 0 0 2px #FFFFFF;}

/* forecast panel */
.tp-call {border-radius: 12px; padding: 20px 22px 18px; border: 1px solid #DCE2EA; background: #FFFFFF;}
.tp-call.up {background: #EEF8F2; border-color: #C7E7D4;}
.tp-call.down {background: #FDF0EE; border-color: #F2CFCA;}
.tp-call-h {font-size: .9rem; font-weight: 600;}
.tp-call-d {font-size: .8rem; opacity: .7;}
.tp-verdict {display: flex; flex-wrap: wrap; align-items: baseline; column-gap: 14px; row-gap: 6px; margin: 12px 0 2px;}
.tp-verdict-word {font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 600; font-size: 4.2rem; line-height: .9; white-space: nowrap;}
.tp-verdict-word small {font-size: .42em; vertical-align: .55em; margin-right: .12em;}
.tp-verdict-prob {font-size: 1.02rem; line-height: 1.35; flex: 1 1 8.5rem; min-width: 8.5rem;}
.tp-verdict-prob b {font-size: 1.5rem; font-weight: 600; display: block;}
.tp-gauge {margin: 18px 0 6px;}
.tp-track {position: relative; height: 10px; border-radius: 5px;
  background: linear-gradient(90deg, rgba(194,58,47,.22) 0 50%, rgba(18,128,79,.22) 50% 100%);}
.tp-track .fill {position: absolute; top: 0; bottom: 0; border-radius: 5px;}
.tp-track .mid {position: absolute; left: 50%; top: -4px; bottom: -4px; width: 1px; background: #14213D; opacity: .45;}
.tp-track .dot {position: absolute; top: 50%; width: 16px; height: 16px; border-radius: 50%;
  transform: translate(-50%, -50%); box-shadow: 0 0 0 3px #FFFFFF;}
.tp-scale {display: flex; justify-content: space-between; font-size: .76rem; opacity: .7; margin-top: 7px;}
.tp-conv {font-size: .86rem; margin: 10px 0 16px;}
.tp-rec {border-top: 1px solid rgba(20,33,61,.12); padding-top: 12px;}
.tp-rec-h {font-size: .86rem; font-weight: 600; margin-bottom: 6px;}
.tp-row {display: flex; justify-content: space-between; gap: 12px; font-size: .88rem; padding: 4px 0;}
.tp-row span {opacity: .75;}
.tp-row b {font-weight: 500; font-variant-numeric: tabular-nums; text-align: right;}
.tp-note {font-size: .84rem; margin: 10px 0 0; padding: 9px 11px; border-radius: 8px; background: rgba(255,255,255,.75); line-height: 1.45;}
.tp-note.plain {background: #FFFFFF; border: 1px solid #DCE2EA;}

/* sections, tables */
.tp-sec {font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 600; font-size: 1.25rem; margin: 2px 0 2px;}
.tp-sec-sub {font-size: .86rem; opacity: .7; margin-bottom: 8px;}
.tp-table {width: 100%; border-collapse: collapse; font-size: .88rem; font-variant-numeric: tabular-nums;}
.tp-table th {text-align: right; font-weight: 500; font-size: .78rem; opacity: .72; padding: 6px 10px; border-bottom: 1px solid rgba(20,33,61,.16);}
.tp-table td {text-align: right; padding: 7px 10px; border-bottom: 1px solid rgba(20,33,61,.07);}
.tp-table th:first-child, .tp-table td:first-child {text-align: left; padding-left: 0;}
.tp-table .l {text-align: left;}
.tp-cm th, .tp-cm td {white-space: nowrap;}
.tp-cm td.hit {font-weight: 600;}
.tp-swatch {display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 8px; vertical-align: 0;}
.tp-pos {display: inline-block; width: 70px; height: 5px; border-radius: 3px; position: relative; vertical-align: middle;
  background: linear-gradient(90deg, #E9A39B, #E7D9A6 50%, #9FD4B5);}
.tp-pos i {position: absolute; top: 50%; width: 9px; height: 9px; border-radius: 50%; background: #14213D; transform: translate(-50%, -50%);}

/* board tiles (compare) */
.tp-board {display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; margin: 6px 0 20px;}
.tp-board .cell {border-radius: 10px; padding: 12px 14px; border: 1px solid rgba(20,33,61,.08);}
.tp-board .s {font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 600; font-size: 1.15rem;}
.tp-board .n {font-size: .76rem; opacity: .72; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.tp-board .p {font-size: 1.02rem; font-weight: 500; margin-top: 8px;}
.tp-board .c {font-size: .86rem; font-weight: 600;}

/* game + learn */
.tp-reveal {border-radius: 12px; padding: 16px 18px; margin: 6px 0 12px;}
.tp-reveal.win {background: #E8F6EE; border: 1px solid #C7E7D4;}
.tp-reveal.lose {background: #FCECEA; border: 1px solid #F2CFCA;}
.tp-reveal h3 {font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 600; font-size: 1.6rem; margin: 0 0 4px; padding: 0;}
.tp-reveal p {margin: 2px 0; font-size: .92rem;}
.tp-card {background: #FFFFFF; border: 1px solid #DCE2EA; border-radius: 12px; padding: 16px 18px; line-height: 1.55; font-size: .93rem;}
.tp-card h4 {font-size: 1rem; font-weight: 600; margin: 0 0 6px; padding: 0;}
.tp-card ul {margin: 6px 0 0 18px; padding: 0;} .tp-card li {margin: 3px 0;}
.tp-reading {border-radius: 12px; padding: 14px 16px; font-size: .93rem; line-height: 1.5;}
.tp-wf {display: grid; gap: 7px; margin: 8px 0 6px;}
.tp-wf .r {display: grid; grid-template-columns: 64px 1fr; gap: 10px; align-items: center; font-size: .8rem;}
.tp-wf .bar {display: flex; height: 18px; border-radius: 5px; overflow: hidden; background: #E9EDF3;}
.tp-wf .tr {background: #3355D6;} .tp-wf .te {background: #eda100;}
.tp-legend {display: flex; gap: 18px; font-size: .8rem; opacity: .85; margin-top: 4px;}

.tp-foot {font-size: .8rem; opacity: .62; margin-top: 28px; padding-top: 14px; border-top: 1px solid rgba(20,33,61,.12);}

/* coloured game buttons */
.st-key-guess_up button {background: #12804F; border-color: #12804F; color: #FFFFFF;}
.st-key-guess_up button:hover {background: #0E6B42; border-color: #0E6B42; color: #FFFFFF;}
.st-key-guess_down button {background: #C23A2F; border-color: #C23A2F; color: #FFFFFF;}
.st-key-guess_down button:hover {background: #A8322A; border-color: #A8322A; color: #FFFFFF;}
.st-key-guess_up button p, .st-key-guess_down button p {font-size: 1.05rem; font-weight: 600;}

@media (max-width: 1000px) {
  .tp-tiles {grid-template-columns: repeat(3, minmax(0, 1fr));}
  .tp-hero {grid-template-columns: minmax(0, 1fr) auto;} .tp-hero .spark {display: none;}
}
@media (max-width: 640px) {
  .tp-hero {grid-template-columns: 1fr; padding: 18px;} .tp-hero .sym {font-size: 2.5rem;} .tp-hero .px {font-size: 2rem;}
  .tp-hero .quote {text-align: left;} .tp-hero .chips {align-items: flex-start;}
  .tp-tiles {grid-template-columns: repeat(2, minmax(0, 1fr));}
  .tp-tile .val {white-space: normal;} .tp-verdict-word {font-size: 3.2rem;}
}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


# ---------------------------------------------------------------- formatting
def esc(text) -> str:
    return html.escape(str(text))


def isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def money(x: float, currency: str, digits: int = 2) -> str:
    if isnan(x):
        return "n/a"
    sign = "−" if x < 0 else ""
    x = abs(x)
    if currency == "GBp":
        return f"{sign}{x:,.{digits}f}p"
    sym = CURRENCY_SYMBOLS.get(currency)
    if sym:
        return f"{sign}{sym}{x:,.{digits}f}"
    return f"{sign}{x:,.{digits}f} {currency}".strip()


def num(x: float, digits: int = 2, signed: bool = False) -> str:
    if isnan(x):
        return "n/a"
    s = f"{abs(x):,.{digits}f}"
    if x < 0 and s.strip("0.,") != "":
        return "−" + s
    return ("+" + s) if signed and x > 0 else s


def pct(x: float, digits: int = 1, signed: bool = False) -> str:
    return "n/a" if isnan(x) else num(x * 100, digits, signed) + "%"


def big(n: float) -> str:
    if isnan(n) or n == 0:
        return "n/a"
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:,.2f}{unit}"
    return f"{n:,.0f}"


def day(d: pd.Timestamp, weekday: bool = False) -> str:
    return (f"{d:%a}, " if weekday else "") + f"{d.day} {d:%b %Y}"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def direction_class(x: float) -> str:
    return "tp-up" if x > 0 else "tp-down" if x < 0 else ""


def tone(x: float) -> str:
    return "up" if x > 0 else "down" if x < 0 else ""


def arrow(x: float) -> str:
    return "▲" if x > 0 else "▼" if x < 0 else "■"


# ---------------------------------------------------------------- data
@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbol: str):
    """Quoted prices with chart indicators, metadata, and whether today's session is still open."""
    prices, meta = load_history(symbol, "10y")
    return add_indicators(prices), meta, session_in_progress(prices, meta)


@st.cache_data(ttl=900, show_spinner=False)
def load_model(symbol: str):
    """Everything above plus model features and the walk-forward analysis (cached 15 minutes)."""
    prices, meta, live = load_prices(symbol)
    data = add_features(prices)
    if data.empty:
        raise DataError(f"{symbol} has too little price history to compute indicators.")
    try:
        result, model_error = analyze(data, live_last_bar=live), None
    except ValueError as exc:
        result, model_error = None, str(exc)
    return prices, meta, data, live, result, model_error


def current_symbol() -> str:
    if "symbol" not in st.session_state:
        st.session_state.symbol = (st.query_params.get("symbol") or "AAPL").strip().upper()
    return st.session_state.symbol


def _symbol_typed() -> None:
    value = (st.session_state.get("_sym_input") or "").strip().upper()
    if value:
        st.session_state.symbol = value


def _symbol_picked() -> None:
    value = st.session_state.get("_sym_pill")
    if value:
        st.session_state.symbol = value


def symbol_picker() -> str:
    """Symbol box plus one-click popular symbols. The choice is shared by every page."""
    sym = current_symbol()
    st.session_state._sym_input = sym
    st.session_state._sym_pill = sym if sym in POPULAR else None
    c1, c2 = st.columns([1, 3.2], vertical_alignment="bottom")
    with c1:
        st.text_input(
            "Symbol", key="_sym_input", on_change=_symbol_typed, placeholder="e.g. MSFT or TCS.NS",
            help="Any Yahoo Finance symbol. NSE stocks end in .NS, BSE in .BO, indices start with ^ (e.g. ^NSEI).",
        )
    with c2:
        st.pills("Popular symbols", POPULAR, key="_sym_pill", on_change=_symbol_picked,
                 label_visibility="collapsed")
    st.query_params["symbol"] = sym
    return sym


def load_model_or_stop(symbol: str):
    try:
        with st.spinner(f"Loading {symbol} and testing the model on its history…"):
            out = load_model(symbol)
    except DataError as exc:
        st.error(str(exc))
        st.stop()
    if len(out[0]) < 2:
        st.error(f"Only one session of data for {symbol}; nothing to compare yet.")
        st.stop()
    return out


def load_prices_or_stop(symbol: str):
    try:
        with st.spinner(f"Loading {symbol} prices…"):
            return load_prices(symbol)
    except DataError as exc:
        st.error(str(exc))
        st.stop()


# ---------------------------------------------------------------- html blocks
def page_title(title: str, sub: str) -> None:
    st.html(f'<div class="tp tp-page"><h1>{esc(title)}</h1><p>{esc(sub)}</p></div>')


def section(title: str, sub: str = "") -> None:
    st.html(f'<div class="tp"><div class="tp-sec">{esc(title)}</div>'
            + (f'<div class="tp-sec-sub">{esc(sub)}</div>' if sub else "") + "</div>")


def footer() -> None:
    st.html(
        '<div class="tp tp-foot">Prices from Yahoo Finance via yfinance, refreshed every 15 minutes. '
        "Displayed prices are as quoted (split-adjusted); the model uses dividend-adjusted prices. "
        "Educational project, not investment advice.</div>"
    )


def sparkline_svg(values: pd.Series, color: str) -> str:
    v = values.dropna().to_numpy()
    if len(v) < 2:
        return ""
    lo, hi = v.min(), v.max()
    span = hi - lo or 1
    w, h = 300, 64
    xs = np.linspace(0, w, len(v))
    ys = h - 4 - (v - lo) / span * (h - 8)
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"0,{h} " + line + f" {w},{h}"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
           f'<defs><linearGradient id="sg" x1="0" x2="0" y1="0" y2="1">'
           f'<stop offset="0" stop-color="{color}" stop-opacity=".35"/>'
           f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
           f'<polygon points="{area}" fill="url(#sg)"/>'
           f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" '
           f'stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>')
    # st.html strips inline <svg>, so the sparkline is embedded as an image.
    data = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}" alt="" style="width:100%;height:64px;display:block">'


def hero_html(meta: dict, prices: pd.DataFrame, live: bool) -> str:
    last, prev = prices["Close"].iloc[-1], prices["Close"].iloc[-2]
    change, change_pct = last - prev, last / prev - 1
    cur = meta["currency"]
    where = ", ".join(p for p in [meta["exchange"], f"prices in {cur}" if cur else ""] if p)
    when = day(prices.index[-1], weekday=True)
    tail = prices["Close"].iloc[-66:]
    spark_color = "#6BE7A5" if tail.iloc[-1] >= tail.iloc[0] else "#FF9487"
    spark_change = tail.iloc[-1] / tail.iloc[0] - 1
    if live:
        t = meta.get("last_trade_time")
        stamp = f" as of {t.tz_convert(meta['timezone']):%H:%M %Z}" if t is not None else ""
        live_chip = f'<span class="tp-chip live">Market open, live price{esc(stamp)}</span>'
        label = "today"
    else:
        live_chip, label = "", f"on {day(prices.index[-1])}"
    chip_cls = tone(change) or "flat"
    return f"""
<div class="tp tp-hero">
  <div>
    <div class="sym">{esc(meta['symbol'])}</div>
    <div class="name">{esc(meta['name'])}</div>
    <div class="sub">{esc(where)}. Last session {esc(when)}.</div>
  </div>
  <div class="spark" title="Last 3 months: {esc(pct(spark_change, 1, signed=True))}">
    {sparkline_svg(tail, spark_color)}
    <div class="sub" style="margin-top:4px">3 months {esc(pct(spark_change, 1, signed=True))}</div>
  </div>
  <div class="quote">
    <div class="px">{esc(money(last, cur))}</div>
    <div class="chips">
      <span class="tp-chip {chip_cls}">{arrow(change)} {esc(num(change, 2, signed=True))} ({esc(pct(change_pct, 2, signed=True))}) {esc(label)}</span>
      {live_chip}
    </div>
  </div>
</div>"""


def tiles_html(cells: list[tuple], cols: int | None = None) -> str:
    """cells: (label, value, note, tone) where note may be pre-built HTML starting with '<'."""
    out = [f'<div class="tp tp-tiles" style="--cols:{cols or len(cells)}">']
    for label, value, note, cls in cells:
        note_html = note if str(note).startswith("<") else esc(note)
        out.append(
            f'<div class="tp-tile {cls}"><div class="label">{esc(label)}</div>'
            f'<div class="val">{esc(value)}</div><div class="note">{note_html}</div></div>'
        )
    out.append("</div>")
    return "".join(out)


def quote_tiles(meta: dict, prices: pd.DataFrame, live: bool) -> str:
    cur = meta["currency"]
    last = prices.iloc[-1]
    year = prices[prices.index > prices.index[-1] - pd.DateOffset(years=1)]
    lo52, hi52 = year["Low"].min(), year["High"].max()
    pos = 0.5 if hi52 == lo52 else (last["Close"] - lo52) / (hi52 - lo52)
    off_high = last["Close"] / hi52 - 1

    vol20 = np.log(prices["Adj Close"]).diff().iloc[-20:].std() * math.sqrt(252)
    vol_tone = "amber" if vol20 >= 0.45 else ""
    vol_note = "High for a stock" if vol20 >= 0.45 else "20-day, annualised"

    rsi = prices["RSI14"].iloc[-1]
    if rsi >= 70:
        rsi_note, rsi_tone = "Overbought (above 70)", "amber"
    elif rsi <= 30:
        rsi_note, rsi_tone = "Oversold (below 30)", "cobalt"
    else:
        rsi_note, rsi_tone = "Neutral zone (30 to 70)", ""

    volume = prices["Volume"].where(prices["Volume"] > 0)
    avg_vol = volume.iloc[-21:-1].mean()
    ratio = last["Volume"] / avg_vol if avg_vol and not isnan(avg_vol) else math.nan
    if isnan(ratio):
        vol_note2, vol_tone2 = "No volume reported", ""
    else:
        vol_note2 = f"{'So far today' if live else 'Last session'} {num(ratio, 2)}× average"
        vol_tone2 = "cobalt" if ratio >= 1.5 and not live else ""

    change = last["Close"] / prices["Close"].iloc[-2] - 1
    return tiles_html([
        ("Previous close", money(prices["Close"].iloc[-2], cur), f"Open {money(last['Open'], cur)}", ""),
        ("Day change", pct(change, 2, signed=True), f"Range {money(last['Low'], cur)} – {num(last['High'])}",
         tone(change)),
        ("52-week range", f"{money(lo52, cur)} – {num(hi52)}",
         f'<div class="tp-range"><i style="left:{pos * 100:.1f}%"></i></div>{esc(pct(off_high, 1))} from high', ""),
        ("Volatility", pct(vol20, 1), vol_note, vol_tone),
        ("RSI (14)", num(rsi, 1), rsi_note, rsi_tone),
        ("Average volume", big(avg_vol), vol_note2, vol_tone2),
    ])


def call_html(res, symbol: str, live: bool) -> str:
    p = res.prob_up
    up = res.call == "Up"
    color = UP if up else DOWN
    lo, hi = sorted((0.5, p))
    gap_pts = abs(p - 0.5) * 100
    if gap_pts >= CONVICTION * 100:
        conv = f"Higher conviction: {num(gap_pts, 1)} points from a coin flip."
    else:
        conv = f"Low conviction: only {num(gap_pts, 1)} points from a coin flip."
    after = "today's close" if live else f"the {day(res.as_of)} close"

    m = res.metrics
    edge = round((m["accuracy"] - m["baseline"]) * 100, 1)
    if m["accuracy_ci_low"] > m["baseline"]:
        verdict = "Beats the baseline by more than the margin of error on this stock."
    elif edge == 0:
        verdict = "Level with the baseline on this stock. Treat the call as close to a coin flip."
    elif edge > 0:
        verdict = (f"Ahead of the baseline by {num(edge, 1)} points, which is inside the margin of error. "
                   "Treat the call as a slight lean, not a signal.")
    else:
        verdict = (f"Behind the baseline by {num(-edge, 1)} points on this stock. "
                   "Treat the call as close to a coin flip.")

    rows = [
        (f"Accuracy, last {m['test_days']} sessions", pct(m["accuracy"])),
        ("95% confidence range", f"{pct(m['accuracy_ci_low'])} – {pct(m['accuracy_ci_high'])}"),
        (f"Baseline ({m['baseline_label']})", pct(m["baseline"])),
        (f"Calls ≥ {int(CONVICTION * 100)} pts from 50%",
         f"{pct(m['conviction_accuracy'])} right, {m['conviction_n']} calls" if m["conviction_n"] else "none"),
    ]
    rows_html = "".join(f'<div class="tp-row"><span>{esc(a)}</span><b>{esc(b)}</b></div>' for a, b in rows)
    return f"""
<div class="tp tp-call {'up' if up else 'down'}">
  <div class="tp-call-h">Next session</div>
  <div class="tp-call-d">Forecast for the session after {esc(after)}</div>
  <div class="tp-verdict">
    <div class="tp-verdict-word" style="color:{color}"><small>{'▲' if up else '▼'}</small>{res.call}</div>
    <div class="tp-verdict-prob"><b>{esc(pct(p))}</b>chance of a higher close</div>
  </div>
  <div class="tp-gauge" role="img" aria-label="Probability of a higher close: {esc(pct(p))}">
    <div class="tp-track">
      <div class="fill" style="left:{lo * 100:.2f}%; width:{(hi - lo) * 100:.2f}%; background:{color}; opacity:.6"></div>
      <div class="mid"></div>
      <div class="dot" style="left:{p * 100:.2f}%; background:{color}"></div>
    </div>
    <div class="tp-scale"><span>Lower close</span><span>50%</span><span>Higher close</span></div>
  </div>
  <div class="tp-conv">{esc(conv)}</div>
  <div class="tp-rec">
    <div class="tp-rec-h">Track record on {esc(symbol)}</div>
    {rows_html}
    <p class="tp-note">{esc(verdict)}</p>
  </div>
</div>"""


def confusion_html(m: dict) -> str:
    def cell(n, hit):
        return f'<td class="{"hit" if hit else ""}">{n}</td>'
    return f"""
<table class="tp tp-table tp-cm">
  <tr><th></th><th>Went up</th><th>Went down</th><th>Hit rate</th></tr>
  <tr><td>Called Up</td>{cell(m['true_up'], True)}{cell(m['false_up'], False)}<td>{esc(pct(m['precision_up']))}</td></tr>
  <tr><td>Called Down</td>{cell(m['false_down'], False)}{cell(m['true_down'], True)}<td>{esc(pct(m['precision_down']))}</td></tr>
  <tr><td>Total</td><td>{m['true_up'] + m['false_down']}</td><td>{m['false_up'] + m['true_down']}</td><td>{esc(pct(m['accuracy']))}</td></tr>
</table>"""


def feature_fmt(f: str):
    if f in PCT_FEATURES:
        digits = 3 if f == "MACDHist" else 2
        signed = f not in {"Volatility10D", "ATRPct"}
        return lambda v: pct(v, digits, signed=signed)
    if f == "VolumeRatio":
        return lambda v: num(v, 2) + "×"
    if f == "RSI14Adj":
        return lambda v: num(v, 1)
    return lambda v: num(v, 2)


def inputs_html(data: pd.DataFrame) -> str:
    rows = []
    hist = data[FEATURES]
    today = hist.iloc[-1]
    for f in FEATURES:
        series = hist[f].dropna()
        lo, hi = series.quantile(0.05), series.quantile(0.95)
        rank = int(round((series <= today[f]).mean() * 100))
        fmt = feature_fmt(f)
        rows.append(
            f"<tr><td>{esc(FEATURE_LABELS[f])}</td><td><b>{esc(fmt(today[f]))}</b></td>"
            f"<td>{esc(fmt(lo))} to {esc(fmt(hi))}</td><td>{esc(ordinal(max(rank, 1)))}</td></tr>"
        )
    return ('<table class="tp tp-table" style="max-width:920px"><tr><th>Input</th><th>Latest</th>'
            "<th>Usual range (middle 90%)</th><th>Percentile</th></tr>" + "".join(rows) + "</table>")


# ---------------------------------------------------------------- charts
def hover_param():
    return alt.selection_point(encodings=["x"], nearest=True, on="pointerover", clear="pointerout", empty=False)


def date_axis(span_days: int, labels: bool = True) -> alt.Axis:
    if span_days <= 45:
        fmt, ticks = "%d %b", {"interval": "week", "step": 1}
    elif span_days <= 200:
        fmt, ticks = "%d %b", {"interval": "month", "step": 1}
    elif span_days <= 400:
        fmt, ticks = "%b %Y", {"interval": "month", "step": 1}
    elif span_days <= 800:
        fmt, ticks = "%b %Y", {"interval": "month", "step": 3}
    else:
        fmt, ticks = "%Y", {"interval": "year", "step": 1}
    return alt.Axis(format=fmt, tickCount=ticks, labels=labels, ticks=False, title=None, grid=False,
                    labelOverlap=True, labelPadding=6)


def _legend():
    return alt.Legend(orient="top", direction="horizontal", title=None, symbolType="stroke",
                      symbolStrokeWidth=2.5, labelFontSize=12)


def price_chart(view: pd.DataFrame, overlays: list[str], currency: str, height: int = 330,
                marker: pd.Timestamp | None = None):
    """Price panel (close with gradient fill + overlays) and a volume panel coloured by up/down day."""
    span = max((view.index[-1] - view.index[0]).days, 1)
    cols = ["Close"] + overlays
    names = {"Close": "Close", **OVERLAY_LABELS}
    wide = view.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume", "ChangePct", *overlays]].copy()
    wide["Dir"] = np.where(wide["ChangePct"].fillna(0) >= 0, "Up day", "Down day")
    long = wide.melt(id_vars="Date", value_vars=cols, var_name="Key", value_name="Price")
    long["Series"] = long["Key"].map(names)
    domain = [names[c] for c in cols]
    yfmt = ",.0f" if view["Close"].median() >= 1000 else ",.2f"
    lo = float(np.nanmin(wide[cols].to_numpy()))
    hi = float(np.nanmax(wide[cols].to_numpy()))
    pad = (hi - lo) * 0.06 or hi * 0.02
    yscale = alt.Scale(domain=[lo - pad, hi + pad], nice=False, zero=False)

    color = alt.Color("Series:N", scale=alt.Scale(domain=domain, range=[SERIES[c] for c in cols]),
                      legend=_legend() if overlays else None)
    hover = hover_param()
    has_volume = view["Volume"].gt(0).any()
    x_top = alt.X("Date:T", axis=date_axis(span, labels=not has_volume))
    x_bottom = alt.X("Date:T", axis=date_axis(span))

    area = alt.Chart(wide).mark_area(
        clip=True, color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=0, y2=1, stops=[
            alt.GradientStop(color="rgba(42,120,214,0.30)", offset=0),
            alt.GradientStop(color="rgba(42,120,214,0.0)", offset=1)])
    ).encode(x=x_top, y=alt.Y("Close:Q", scale=yscale), y2=alt.datum(lo - pad))
    lines = alt.Chart(long).mark_line(strokeCap="round", strokeJoin="round", clip=True).encode(
        x=x_top,
        y=alt.Y("Price:Q", scale=yscale, title=None,
                axis=alt.Axis(format=yfmt, orient="right", tickCount=6, minExtent=Y_EXTENT, maxExtent=Y_EXTENT)),
        color=color,
        strokeWidth=alt.condition("datum.Series == 'Close'", alt.value(2.2), alt.value(1.3)),
    )
    dots = alt.Chart(long).mark_circle(size=50, stroke="#FFFFFF", strokeWidth=2).encode(
        x=x_top, y=alt.Y("Price:Q", scale=yscale), color=color,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    tooltip = [
        alt.Tooltip("Date:T", title="Date", format="%a %d %b %Y"),
        alt.Tooltip("Close:Q", title=f"Close ({currency})", format=",.2f"),
        alt.Tooltip("ChangePct:Q", title="Change", format="+.2%"),
        alt.Tooltip("Open:Q", format=",.2f"),
        alt.Tooltip("High:Q", format=",.2f"),
        alt.Tooltip("Low:Q", format=",.2f"),
    ] + [alt.Tooltip(f"{c}:Q", title=OVERLAY_LABELS[c], format=",.2f") for c in overlays] + [
        alt.Tooltip("Volume:Q", format=",.0f")]
    rule = alt.Chart(wide).mark_rule(color=RULE, strokeWidth=1).encode(
        x=x_top, opacity=alt.condition(hover, alt.value(0.7), alt.value(0)), tooltip=tooltip).add_params(hover)
    layers = [area, lines, rule, dots]
    if marker is not None:
        layers.append(alt.Chart(pd.DataFrame({"Date": [marker]})).mark_rule(
            color=INK_RULE, strokeDash=[4, 3], strokeWidth=1.2).encode(x="Date:T"))
    top = alt.layer(*layers).properties(height=height)

    if not has_volume:
        return top, None
    vol_hover = hover_param()
    # Bars take 60% of each day's slot, so they scale with the chart width (phones included).
    vol = alt.Chart(wide).mark_bar(width=alt.RelativeBandSize(0.6), opacity=0.55).encode(
        x=alt.X("yearmonthdate(Date):T", axis=date_axis(span), title=None),
        y=alt.Y("Volume:Q", title=None, axis=alt.Axis(format="~s", orient="right", tickCount=2, grid=False,
                                                     minExtent=Y_EXTENT, maxExtent=Y_EXTENT,
                                                     labelExpr="datum.value == 0 ? '' : datum.label")),
        color=alt.Color("Dir:N", scale=alt.Scale(domain=["Up day", "Down day"], range=[UP, DOWN]), legend=None),
        tooltip=[alt.Tooltip("Date:T", format="%a %d %b %Y"), alt.Tooltip("Volume:Q", format=",.0f"),
                 alt.Tooltip("ChangePct:Q", title="Change", format="+.2%")],
    )
    vol_rule = alt.Chart(wide).mark_rule(color=RULE).encode(
        x=x_bottom, opacity=alt.condition(vol_hover, alt.value(0.7), alt.value(0))).add_params(vol_hover)
    return top, alt.layer(vol, vol_rule).properties(height=78)


def show_price_chart(view, overlays, currency, height=330, marker=None) -> None:
    top, volume = price_chart(view, overlays, currency, height=height, marker=marker)
    with st.container(gap=None):
        st.altair_chart(top, width="stretch")
        if volume is not None:
            st.altair_chart(volume, width="stretch")


def lines_chart(df: pd.DataFrame, series: list[str], colors: list[str], height: int = 280,
                value_fmt: str = ".1f", baseline: float | None = None, tooltip_extra: list | None = None,
                shading: pd.DataFrame | None = None) -> alt.LayerChart:
    """Multi-line chart on one axis with a shared hover readout. df: index Date, one column per series."""
    tip_df = df.reset_index().rename(columns={df.index.name or "index": "Date"})
    wide = tip_df[["Date", *series]]
    long = wide.melt(id_vars="Date", var_name="Series", value_name="Value")
    span = max((df.index[-1] - df.index[0]).days, 1)
    x = alt.X("Date:T", axis=date_axis(span))
    color = alt.Color("Series:N", scale=alt.Scale(domain=series, range=colors), legend=_legend())
    dot_color = alt.Color("Series:N", scale=alt.Scale(domain=series, range=colors), legend=None)
    hover = hover_param()
    layers = []
    if shading is not None and len(shading):
        layers.append(alt.Chart(shading).mark_rect(opacity=0.10).encode(
            x="start:T", x2="end:T",
            color=alt.Color("kind:N", scale=alt.Scale(domain=["Long", "Short"], range=[UP, DOWN]), legend=None)))
    if baseline is not None:
        layers.append(alt.Chart(pd.DataFrame({"y": [baseline]})).mark_rule(color=RULE).encode(y="y:Q"))
    layers.append(alt.Chart(long).mark_line(strokeWidth=2, strokeCap="round").encode(
        x=x, y=alt.Y("Value:Q", scale=alt.Scale(zero=False), title=None,
                     axis=alt.Axis(orient="right", tickCount=5, format=value_fmt)),
        color=color))
    tips = [alt.Tooltip("Date:T", format="%a %d %b %Y")] + [
        alt.Tooltip(f"{s}:Q", format=value_fmt) for s in series] + (tooltip_extra or [])
    layers.append(alt.Chart(tip_df).mark_rule(color=RULE).encode(
        x=x, opacity=alt.condition(hover, alt.value(0.7), alt.value(0)), tooltip=tips).add_params(hover))
    layers.append(alt.Chart(long).mark_circle(size=42, stroke="#FFFFFF", strokeWidth=2).encode(
        x=x, y="Value:Q", color=dot_color, opacity=alt.condition(hover, alt.value(1), alt.value(0))))
    return alt.layer(*layers).resolve_scale(color="independent").properties(height=height)


def importance_chart(imp: pd.Series) -> alt.Chart:
    df = imp.rename("Share").reset_index().rename(columns={"index": "Feature"})
    df["Input"] = df["Feature"].map(FEATURE_LABELS)
    return alt.Chart(df).mark_bar(color=COBALT, cornerRadiusEnd=4, height=14).encode(
        x=alt.X("Share:Q", title=None, axis=alt.Axis(format=".0%", tickCount=4)),
        y=alt.Y("Input:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220, ticks=False, domain=False)),
        tooltip=[alt.Tooltip("Input:N"), alt.Tooltip("Share:Q", title="Share of gain", format=".1%")],
    ).properties(height=24 * len(df))


def block_chart(bt: pd.DataFrame) -> alt.LayerChart:
    rows = []
    for _, g in bt.groupby("TrainRows", sort=True):
        acc = g["Correct"].mean()
        rows.append({"Block": f"{g.index[0]:%b}–{g.index[-1]:%b}", "Dates": f"{day(g.index[0])} to {day(g.index[-1])}",
                     "Accuracy": acc, "Sessions": len(g), "Order": g.index[0],
                     "Result": "Above 50%" if acc > 0.5 else "50% or below"})
    df = pd.DataFrame(rows)
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, size=26).encode(
        x=alt.X("Block:N", sort=alt.SortField("Order"), title=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
        y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 1]), title=None, axis=alt.Axis(format=".0%", tickCount=5)),
        color=alt.Color("Result:N", scale=alt.Scale(domain=["Above 50%", "50% or below"], range=[UP, DOWN]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("Dates:N", title="Test block"), alt.Tooltip("Accuracy:Q", format=".1%"),
                 alt.Tooltip("Sessions:Q")],
    )
    coin = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color=INK_RULE, strokeDash=[4, 3]).encode(y="y:Q")
    return alt.layer(bars, coin).properties(height=230)


def rsi_chart(view: pd.DataFrame, height: int = 240) -> alt.LayerChart:
    span = max((view.index[-1] - view.index[0]).days, 1)
    df = view[["RSI14", "Close"]].reset_index()
    hover = hover_param()
    x = alt.X("Date:T", axis=date_axis(span))
    bands = pd.DataFrame({"lo": [70, 0], "hi": [100, 30], "zone": ["Overbought", "Oversold"]})
    band = alt.Chart(bands).mark_rect(opacity=0.12).encode(
        y="lo:Q", y2="hi:Q",
        color=alt.Color("zone:N", scale=alt.Scale(domain=["Overbought", "Oversold"], range=["#eda100", COBALT]),
                        legend=alt.Legend(orient="top", title=None)))
    levels = alt.Chart(pd.DataFrame({"y": [30, 70]})).mark_rule(color=INK_RULE, strokeDash=[4, 3]).encode(y="y:Q")
    line = alt.Chart(df).mark_line(color=SERIES["Close"], strokeWidth=1.8).encode(
        x=x, y=alt.Y("RSI14:Q", scale=alt.Scale(domain=[0, 100]), title=None,
                     axis=alt.Axis(values=[0, 30, 50, 70, 100], orient="right")))
    rule = alt.Chart(df).mark_rule(color=RULE).encode(
        x=x, opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
        tooltip=[alt.Tooltip("Date:T", format="%a %d %b %Y"), alt.Tooltip("RSI14:Q", title="RSI (14)", format=".1f"),
                 alt.Tooltip("Close:Q", format=",.2f")],
    ).add_params(hover)
    return alt.layer(band, levels, line, rule).properties(height=height)
