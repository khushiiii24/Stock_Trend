"""Stock Trend Predictor dashboard.

Run with:  streamlit run dashboard.py
"""
from __future__ import annotations

import html
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from features import (FEATURE_LABELS, FEATURES, DataError, add_features, add_indicators, load_history,
                      session_in_progress)
from train_model import CONVICTION, analyze

st.set_page_config(
    page_title="Stock Trend Predictor",
    page_icon=":material/show_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------- tokens
UP, DOWN = "#12804F", "#C23A2F"
SERIES = {"Close": "#2a78d6", "SMA5": "#eb6834", "SMA10": "#1baf7a", "EMA20": "#eda100"}
OVERLAY_LABELS = {"SMA5": "5-day SMA", "SMA10": "10-day SMA", "EMA20": "20-day EMA"}
BENCHMARK = "#8E98A6"
VOLUME = "#B7C0CB"
RULE = "#8E98A6"
INK_RULE = "#5B6676"

RANGES = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "2Y": pd.DateOffset(years=2),
    "5Y": pd.DateOffset(years=5),
    "10Y": pd.DateOffset(years=10),
}
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
[data-testid="stMainBlockContainer"], .block-container {padding-top: 4.25rem; padding-bottom: 3rem; max-width: 1320px;}
.tp {font-family: "IBM Plex Sans", sans-serif; color: inherit;}
.tp-up {color: #12804F;} .tp-down {color: #C23A2F;}
.tp-muted {opacity: .66;}
.tp-tab {font-variant-numeric: tabular-nums;}

.tp-head {display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; flex-wrap: wrap;
  padding: 6px 0 18px; border-bottom: 1px solid rgba(20,32,48,.14);}
.tp-sym {font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 600; font-size: 3.5rem; line-height: .9; letter-spacing: -.015em;}
.tp-name {font-size: 1.06rem; font-weight: 500; margin-top: 10px;}
.tp-sub {font-size: .86rem; opacity: .66; margin-top: 2px;}
.tp-quote {text-align: right;}
.tp-last {font-size: 2.5rem; font-weight: 500; line-height: 1; letter-spacing: -.01em;}
.tp-chg {font-size: 1rem; font-weight: 500; margin-top: 8px;}
.tp-live {display: inline-block; margin-top: 8px; font-size: .8rem; padding: 2px 8px; border-radius: 999px;
  background: rgba(30,78,140,.10); color: #1E4E8C;}

.tp-strip {display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); margin: 0 0 22px;
  border-bottom: 1px solid rgba(20,32,48,.14);}
.tp-cell {padding: 14px 16px 14px 0; min-width: 0;}
.tp-cell + .tp-cell {padding-left: 16px; border-left: 1px solid rgba(20,32,48,.09);}
.tp-label {font-size: .78rem; opacity: .66; margin-bottom: 4px;}
.tp-val {font-size: 1.14rem; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.tp-note {font-size: .78rem; opacity: .66; margin-top: 3px;}
.tp-range {position: relative; height: 4px; border-radius: 2px; background: rgba(20,32,48,.14); margin: 9px 0 6px;}
.tp-range i {position: absolute; top: 50%; width: 11px; height: 11px; border-radius: 50%; background: #142030;
  transform: translate(-50%, -50%); box-shadow: 0 0 0 2px #F4F6F8;}

.tp-call {background: #FFFFFF; border: 1px solid #D3D9E1; border-radius: 10px; padding: 20px 22px 18px; color: #142030;}
.tp-call-h {font-size: .9rem; font-weight: 500;}
.tp-call-d {font-size: .8rem; opacity: .66;}
.tp-verdict {display: flex; align-items: baseline; gap: 14px; margin: 12px 0 2px;}
.tp-verdict-word {font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 600; font-size: 4.2rem; line-height: .9; white-space: nowrap;}
.tp-verdict-word small {font-size: .42em; vertical-align: .55em; margin-right: .12em;}
.tp-verdict-prob {font-size: 1.02rem; line-height: 1.35;}
.tp-verdict-prob b {font-size: 1.5rem; font-weight: 600; display: block;}
.tp-gauge {margin: 18px 0 6px;}
.tp-track {position: relative; height: 10px; border-radius: 5px;
  background: linear-gradient(90deg, rgba(194,58,47,.16) 0 50%, rgba(18,128,79,.16) 50% 100%);}
.tp-track .fill {position: absolute; top: 0; bottom: 0;}
.tp-track .mid {position: absolute; left: 50%; top: -4px; bottom: -4px; width: 1px; background: #142030; opacity: .45;}
.tp-track .dot {position: absolute; top: 50%; width: 16px; height: 16px; border-radius: 50%;
  transform: translate(-50%, -50%); box-shadow: 0 0 0 3px #FFFFFF;}
.tp-scale {display: flex; justify-content: space-between; font-size: .76rem; opacity: .66; margin-top: 7px;}
.tp-conv {font-size: .86rem; margin: 10px 0 16px;}
.tp-rec {border-top: 1px solid rgba(20,32,48,.12); padding-top: 12px;}
.tp-rec-h {font-size: .86rem; font-weight: 500; margin-bottom: 6px;}
.tp-row {display: flex; justify-content: space-between; gap: 12px; font-size: .88rem; padding: 4px 0;}
.tp-row span {opacity: .72;}
.tp-row b {font-weight: 500; font-variant-numeric: tabular-nums; text-align: right;}
.tp-verdict-note {font-size: .84rem; margin: 10px 0 0; padding: 8px 10px; border-radius: 6px; background: #F4F6F8; line-height: 1.4;}

.tp-sec {font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 600; font-size: 1.25rem; margin: 2px 0 2px;}
.tp-sec-sub {font-size: .86rem; opacity: .66; margin-bottom: 8px;}
.tp-table {width: 100%; border-collapse: collapse; font-size: .88rem; font-variant-numeric: tabular-nums;}
.tp-table th {text-align: right; font-weight: 500; font-size: .78rem; opacity: .7; padding: 6px 10px; border-bottom: 1px solid rgba(20,32,48,.16);}
.tp-table td {text-align: right; padding: 7px 10px; border-bottom: 1px solid rgba(20,32,48,.07);}
.tp-table th:first-child, .tp-table td:first-child {text-align: left; padding-left: 0;}
.tp-cm th, .tp-cm td {white-space: nowrap;}
.tp-cm td.hit {font-weight: 600;}
.tp-foot {font-size: .8rem; opacity: .6; margin-top: 28px; padding-top: 14px; border-top: 1px solid rgba(20,32,48,.12);}

@media (max-width: 1000px) {
  .tp-strip {grid-template-columns: repeat(3, minmax(0, 1fr));}
  .tp-cell:nth-child(4) {border-left: none; padding-left: 0;}
}
@media (max-width: 640px) {
  .tp-sym {font-size: 2.6rem;} .tp-last {font-size: 2rem;} .tp-quote {text-align: left;}
  .tp-val {white-space: normal;} .tp-verdict-word {font-size: 3.2rem;}
  .tp-strip {grid-template-columns: repeat(2, minmax(0, 1fr));}
  .tp-cell, .tp-cell + .tp-cell {border-left: none; padding-left: 0;}
}
</style>
"""


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
    if currency == "GBp":  # London quotes in pence
        return f"{sign}{x:,.{digits}f}p"
    sym = CURRENCY_SYMBOLS.get(currency)
    if sym:
        return f"{sign}{sym}{x:,.{digits}f}"
    return f"{sign}{x:,.{digits}f} {currency}".strip()


def num(x: float, digits: int = 2, signed: bool = False) -> str:
    if isnan(x):
        return "n/a"
    s = f"{abs(x):,.{digits}f}"
    if x < 0:
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


def arrow(x: float) -> str:
    return "▲" if x > 0 else "▼" if x < 0 else "■"


# ---------------------------------------------------------------- data
@st.cache_data(ttl=900, show_spinner=False)
def load(symbol: str):
    """Prices, metadata, features and model results for one symbol (cached 15 minutes)."""
    prices, meta = load_history(symbol, "10y")
    live = session_in_progress(prices, meta)
    display = add_indicators(prices)
    data = add_features(prices)
    if data.empty:
        raise DataError(f"{symbol} has too little price history to compute indicators.")
    try:
        result, model_error = analyze(data, live_last_bar=live), None
    except ValueError as exc:
        result, model_error = None, str(exc)
    return display, meta, data, live, result, model_error


# ---------------------------------------------------------------- charts
def hover_param():
    return alt.selection_point(
        encodings=["x"], nearest=True, on="pointerover", clear="pointerout", empty=False
    )


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


Y_EXTENT = 58  # fixed y-axis width so the price and volume panels line up


def price_chart(view: pd.DataFrame, overlays: list[str], currency: str) -> tuple[alt.LayerChart, alt.LayerChart]:
    span = (view.index[-1] - view.index[0]).days
    cols = ["Close"] + overlays
    names = {"Close": "Close", **OVERLAY_LABELS}
    wide = view.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume", "ChangePct", *overlays]]
    long = wide.melt(id_vars="Date", value_vars=cols, var_name="Key", value_name="Price")
    long["Series"] = long["Key"].map(names)
    domain = [names[c] for c in cols]
    yfmt = ",.0f" if view["Close"].median() >= 1000 else ",.2f"

    color = alt.Color(
        "Series:N",
        scale=alt.Scale(domain=domain, range=[SERIES[c] for c in cols]),
        legend=alt.Legend(orient="top", direction="horizontal", title=None, symbolType="stroke",
                          symbolStrokeWidth=2.5, labelFontSize=12) if overlays else None,
    )
    hover = hover_param()
    has_volume = view["Volume"].gt(0).any()
    x_top = alt.X("Date:T", axis=date_axis(span, labels=not has_volume))
    x_bottom = alt.X("Date:T", axis=date_axis(span))

    lines = (
        alt.Chart(long)
        .mark_line(strokeCap="round", strokeJoin="round")
        .encode(
            x=x_top,
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False, nice=True), title=None,
                    axis=alt.Axis(format=yfmt, orient="right", tickCount=6, minExtent=Y_EXTENT, maxExtent=Y_EXTENT)),
            color=color,
            strokeWidth=alt.condition("datum.Series == 'Close'", alt.value(2), alt.value(1.25)),
        )
    )
    dots = (
        alt.Chart(long)
        .mark_circle(size=46, stroke="#FFFFFF", strokeWidth=2)
        .encode(x=x_top, y="Price:Q", color=color, opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    )
    tooltip = [
        alt.Tooltip("Date:T", title="Date", format="%a %d %b %Y"),
        alt.Tooltip("Close:Q", title=f"Close ({currency})", format=",.2f"),
        alt.Tooltip("ChangePct:Q", title="Change", format="+.2%"),
        alt.Tooltip("Open:Q", format=",.2f"),
        alt.Tooltip("High:Q", format=",.2f"),
        alt.Tooltip("Low:Q", format=",.2f"),
    ] + [alt.Tooltip(f"{c}:Q", title=OVERLAY_LABELS[c], format=",.2f") for c in overlays] + [
        alt.Tooltip("Volume:Q", format=",.0f"),
    ]
    rule = (
        alt.Chart(wide)
        .mark_rule(color=RULE, strokeWidth=1)
        .encode(x=x_top, opacity=alt.condition(hover, alt.value(0.7), alt.value(0)), tooltip=tooltip)
        .add_params(hover)
    )
    top = alt.layer(lines, rule, dots).properties(height=330)

    bar_width = max(1.0, min(7.0, 0.62 * 900 / max(len(view), 1)))
    vol = (
        alt.Chart(wide)
        .mark_bar(color=VOLUME, size=bar_width)
        .encode(x=x_bottom, y=alt.Y("Volume:Q", title=None,
                                    axis=alt.Axis(format="~s", orient="right", tickCount=2, grid=False,
                                                  minExtent=Y_EXTENT, maxExtent=Y_EXTENT,
                                                  labelExpr="datum.value == 0 ? '' : datum.label")))
    )
    vol_hover = hover_param()
    vol_rule = alt.Chart(wide).mark_rule(color=RULE).encode(
        x=x_bottom, opacity=alt.condition(vol_hover, alt.value(0.7), alt.value(0)),
        tooltip=[alt.Tooltip("Date:T", format="%a %d %b %Y"), alt.Tooltip("Volume:Q", format=",.0f")],
    ).add_params(vol_hover)
    bottom = alt.layer(vol, vol_rule).properties(height=78) if has_volume else None
    return top, bottom


def equity_chart(bt: pd.DataFrame) -> alt.LayerChart:
    df = bt[["ModelEquity", "BuyHoldEquity", "ProbUp", "NextReturn"]].reset_index()
    long = df.melt(id_vars="Date", value_vars=["ModelEquity", "BuyHoldEquity"], var_name="Key", value_name="Value")
    long["Series"] = long["Key"].map({"ModelEquity": "Following the model", "BuyHoldEquity": "Buy and hold"})
    domain = ["Following the model", "Buy and hold"]
    color = alt.Color("Series:N", scale=alt.Scale(domain=domain, range=[SERIES["Close"], BENCHMARK]),
                      legend=alt.Legend(orient="top", direction="horizontal", title=None,
                                        symbolType="stroke", symbolStrokeWidth=2.5, labelFontSize=12))
    hover = hover_param()
    x = alt.X("Date:T", axis=date_axis((bt.index[-1] - bt.index[0]).days))
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=x, y=alt.Y("Value:Q", scale=alt.Scale(zero=False), title=None, axis=alt.Axis(orient="right", tickCount=5)),
        color=color,
    )
    base = alt.Chart(pd.DataFrame({"y": [100]})).mark_rule(color=RULE, strokeWidth=1).encode(y="y:Q")
    rule = alt.Chart(df).mark_rule(color=RULE).encode(
        x=x, opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
        tooltip=[
            alt.Tooltip("Date:T", format="%a %d %b %Y"),
            alt.Tooltip("ModelEquity:Q", title="Following the model", format=".1f"),
            alt.Tooltip("BuyHoldEquity:Q", title="Buy and hold", format=".1f"),
            alt.Tooltip("ProbUp:Q", title="P(up) that day", format=".1%"),
            alt.Tooltip("NextReturn:Q", title="Next-day return", format="+.2%"),
        ],
    ).add_params(hover)
    dots = alt.Chart(long).mark_circle(size=40, stroke="#FFFFFF", strokeWidth=2).encode(
        x=x, y="Value:Q", color=color,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    )
    return alt.layer(base, lines, rule, dots).properties(height=280)


def importance_chart(imp: pd.Series) -> alt.Chart:
    df = imp.rename("Share").reset_index().rename(columns={"index": "Feature"})
    df["Input"] = df["Feature"].map(FEATURE_LABELS)
    return (
        alt.Chart(df)
        .mark_bar(color=SERIES["Close"], cornerRadiusEnd=3, height=14)
        .encode(
            x=alt.X("Share:Q", title=None, axis=alt.Axis(format=".0%", tickCount=4)),
            y=alt.Y("Input:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220, ticks=False, domain=False)),
            tooltip=[alt.Tooltip("Input:N"), alt.Tooltip("Share:Q", title="Share of gain", format=".1%")],
        )
        .properties(height=24 * len(df))
    )


def block_chart(bt: pd.DataFrame) -> alt.LayerChart:
    rows = []
    for _, g in bt.groupby("TrainRows", sort=True):
        rows.append({
            "Block": f"{g.index[0]:%b}–{g.index[-1]:%b}",
            "Dates": f"{day(g.index[0])} to {day(g.index[-1])}",
            "Accuracy": g["Correct"].mean(),
            "Sessions": len(g),
            "Order": g.index[0],
        })
    df = pd.DataFrame(rows)
    bars = alt.Chart(df).mark_bar(color=SERIES["Close"], cornerRadiusEnd=3, size=26).encode(
        x=alt.X("Block:N", sort=alt.SortField("Order"), title=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
        y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 1]), title=None, axis=alt.Axis(format=".0%", tickCount=5)),
        tooltip=[alt.Tooltip("Dates:N", title="Test block"), alt.Tooltip("Accuracy:Q", format=".1%"),
                 alt.Tooltip("Sessions:Q")],
    )
    coin = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color=INK_RULE, strokeDash=[4, 3]).encode(y="y:Q")
    return alt.layer(bars, coin).properties(height=230)


def rsi_chart(view: pd.DataFrame) -> alt.LayerChart:
    span = (view.index[-1] - view.index[0]).days
    df = view[["RSI14", "Close"]].reset_index()
    hover = hover_param()
    x = alt.X("Date:T", axis=date_axis(span))
    band = alt.Chart(pd.DataFrame({"lo": [30], "hi": [70]})).mark_rect(color=RULE, opacity=0.10).encode(
        y="lo:Q", y2="hi:Q")
    levels = alt.Chart(pd.DataFrame({"y": [30, 70]})).mark_rule(color=INK_RULE, strokeDash=[4, 3]).encode(y="y:Q")
    line = alt.Chart(df).mark_line(color=SERIES["Close"], strokeWidth=1.75).encode(
        x=x, y=alt.Y("RSI14:Q", scale=alt.Scale(domain=[0, 100]), title=None,
                     axis=alt.Axis(values=[0, 30, 50, 70, 100], orient="right")))
    rule = alt.Chart(df).mark_rule(color=RULE).encode(
        x=x, opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
        tooltip=[alt.Tooltip("Date:T", format="%a %d %b %Y"), alt.Tooltip("RSI14:Q", title="RSI (14)", format=".1f"),
                 alt.Tooltip("Close:Q", format=",.2f")],
    ).add_params(hover)
    return alt.layer(band, levels, line, rule).properties(height=240)



# ---------------------------------------------------------------- html blocks
def header_html(meta: dict, prices: pd.DataFrame, live: bool) -> str:
    last, prev = prices["Close"].iloc[-1], prices["Close"].iloc[-2]
    change, change_pct = last - prev, last / prev - 1
    cur = meta["currency"]
    where = ", ".join(p for p in [meta["exchange"], f"prices in {cur}" if cur else ""] if p)
    when = day(prices.index[-1], weekday=True)
    if live:
        t = meta.get("last_trade_time")
        stamp = f" as of {t.tz_convert(meta['timezone']):%H:%M %Z}" if t is not None else ""
        tag = f'<div class="tp-live">Market open, live price{esc(stamp)}</div>'
        label = "today"
    else:
        tag, label = "", f"on {day(prices.index[-1])}"
    return f"""
<div class="tp tp-head">
  <div>
    <div class="tp-sym">{esc(meta['symbol'])}</div>
    <div class="tp-name">{esc(meta['name'])}</div>
    <div class="tp-sub">{esc(where)}. Last session {esc(when)}.</div>
  </div>
  <div class="tp-quote">
    <div class="tp-last">{esc(money(last, cur))}</div>
    <div class="tp-chg {direction_class(change)}">{arrow(change)} {esc(num(change, 2, signed=True))} ({esc(pct(change_pct, 2, signed=True))}) {esc(label)}</div>
    {tag}
  </div>
</div>"""


def strip_html(meta: dict, prices: pd.DataFrame, data: pd.DataFrame, live: bool) -> str:
    cur = meta["currency"]
    last = prices.iloc[-1]
    year = prices[prices.index > prices.index[-1] - pd.DateOffset(years=1)]
    lo52, hi52 = year["Low"].min(), year["High"].max()
    pos = 0.5 if hi52 == lo52 else (last["Close"] - lo52) / (hi52 - lo52)
    off_high = last["Close"] / hi52 - 1

    log_ret = np.log(prices["Adj Close"]).diff().iloc[-20:]
    vol20 = log_ret.std() * math.sqrt(252)

    rsi = data["RSI14"].iloc[-1]
    rsi_note = "Overbought (above 70)" if rsi >= 70 else "Oversold (below 30)" if rsi <= 30 else "Neutral zone (30 to 70)"

    volume = prices["Volume"].where(prices["Volume"] > 0)
    avg_vol = volume.iloc[-21:-1].mean()
    today_ratio = last["Volume"] / avg_vol if avg_vol and not isnan(avg_vol) else math.nan
    vol_note = (f"{'So far today' if live else 'Last session'} {num(today_ratio, 2)}× average"
                if not isnan(today_ratio) else "No volume reported")

    cells = [
        ("Previous close", money(prices["Close"].iloc[-2], cur), f"Open {money(last['Open'], cur)}"),
        ("Day range", f"{money(last['Low'], cur)} – {num(last['High'])}",
         f"Spread {pct(last['High'] / last['Low'] - 1, 2)}"),
        ("52-week range", f"{money(lo52, cur)} – {num(hi52)}",
         f'<div class="tp-range"><i style="left:{pos * 100:.1f}%"></i></div>{esc(pct(off_high, 1))} from high'),
        ("Volatility", pct(vol20, 1), "20-day, annualised"),
        ("RSI (14)", num(rsi, 1), rsi_note),
        ("Average volume", big(avg_vol), vol_note),
    ]
    out = ['<div class="tp tp-strip">']
    for label, value, note in cells:
        note_html = note if note.startswith("<div") else esc(note)
        out.append(
            f'<div class="tp-cell"><div class="tp-label">{esc(label)}</div>'
            f'<div class="tp-val">{esc(value)}</div><div class="tp-note">{note_html}</div></div>'
        )
    out.append("</div>")
    return "".join(out)


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
    edge = (m["accuracy"] - m["baseline"]) * 100
    if m["accuracy_ci_low"] > m["baseline"]:
        verdict = "Beats the baseline by more than the margin of error on this stock."
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
<div class="tp tp-call">
  <div class="tp-call-h">Next session</div>
  <div class="tp-call-d">Forecast for the session after {esc(after)}</div>
  <div class="tp-verdict">
    <div class="tp-verdict-word" style="color:{color}"><small>{'▲' if up else '▼'}</small>{res.call}</div>
    <div class="tp-verdict-prob"><b>{esc(pct(p))}</b>chance of a higher close</div>
  </div>
  <div class="tp-gauge" role="img" aria-label="Probability of a higher close: {esc(pct(p))}">
    <div class="tp-track">
      <div class="fill" style="left:{lo * 100:.2f}%; width:{(hi - lo) * 100:.2f}%; background:{color}; opacity:.55"></div>
      <div class="mid"></div>
      <div class="dot" style="left:{p * 100:.2f}%; background:{color}"></div>
    </div>
    <div class="tp-scale"><span>Lower close</span><span>50%</span><span>Higher close</span></div>
  </div>
  <div class="tp-conv">{esc(conv)}</div>
  <div class="tp-rec">
    <div class="tp-rec-h">Track record on {esc(symbol)}</div>
    {rows_html}
    <p class="tp-verdict-note">{esc(verdict)}</p>
  </div>
</div>"""


def stats_strip(cells: list[tuple[str, str, str]]) -> str:
    out = ['<div class="tp tp-strip" style="grid-template-columns: repeat(%d, minmax(0, 1fr))">' % len(cells)]
    for label, value, note in cells:
        out.append(
            f'<div class="tp-cell"><div class="tp-label">{esc(label)}</div>'
            f'<div class="tp-val">{esc(value)}</div><div class="tp-note">{esc(note)}</div></div>'
        )
    out.append("</div>")
    return "".join(out)


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


def inputs_html(data: pd.DataFrame) -> str:
    rows = []
    hist = data[FEATURES]
    today = hist.iloc[-1]
    for f in FEATURES:
        series = hist[f].dropna()
        lo, hi = series.quantile(0.05), series.quantile(0.95)
        rank = int(round((series <= today[f]).mean() * 100))
        if f in PCT_FEATURES:
            digits = 3 if f == "MACDHist" else 2
            signed = f not in {"Volatility10D", "ATRPct"}
            fmt = lambda v, d=digits, s=signed: pct(v, d, signed=s)
        elif f == "VolumeRatio":
            fmt = lambda v: num(v, 2) + "×"
        elif f == "RSI14Adj":
            fmt = lambda v: num(v, 1)
        else:
            fmt = lambda v: num(v, 2)
        rows.append(
            f"<tr><td>{esc(FEATURE_LABELS[f])}</td><td><b>{esc(fmt(today[f]))}</b></td>"
            f"<td>{esc(fmt(lo))} to {esc(fmt(hi))}</td><td>{esc(ordinal(max(rank, 1)))}</td></tr>"
        )
    return (
        '<table class="tp tp-table" style="max-width:920px"><tr><th>Input</th><th>Latest</th><th>Usual range (middle 90%)</th>'
        "<th>Percentile</th></tr>" + "".join(rows) + "</table>"
    )


def section(title: str, sub: str = "") -> None:
    st.html(f'<div class="tp"><div class="tp-sec">{esc(title)}</div>'
            + (f'<div class="tp-sec-sub">{esc(sub)}</div>' if sub else "") + "</div>")


# ---------------------------------------------------------------- page
st.html(CSS)

default_symbol = st.query_params.get("symbol", "AAPL")
c_in, c_hint = st.columns([1, 3], vertical_alignment="bottom")
with c_in:
    symbol = st.text_input(
        "Symbol",
        value=default_symbol,
        placeholder="e.g. MSFT or TCS.NS",
        help="Any Yahoo Finance symbol. NSE stocks end in .NS, BSE in .BO, indices start with ^ (e.g. ^NSEI).",
    ).strip().upper()
with c_hint:
    st.caption("Try MSFT, NVDA, RELIANCE.NS, TCS.NS or ^NSEI. Prices refresh every 15 minutes.")

if not symbol:
    st.info("Enter a symbol to load its prices and forecast.")
    st.stop()
st.query_params["symbol"] = symbol

try:
    with st.spinner(f"Loading {symbol} and testing the model on its history…"):
        prices, meta, data, live, res, model_error = load(symbol)
except DataError as exc:
    st.error(str(exc))
    st.stop()

if len(prices) < 2:
    st.error(f"Only one session of data for {symbol}; nothing to compare yet.")
    st.stop()

cur = meta["currency"]
st.html(header_html(meta, prices, live))
st.html(strip_html(meta, prices, data, live))

left, right = st.columns([2.1, 1], gap="large")

with left:
    c1, c2 = st.columns([1.15, 1], vertical_alignment="bottom")
    with c1:
        rng = st.segmented_control("Range", list(RANGES), default="1Y", key="range") or "1Y"
    with c2:
        overlays = st.segmented_control(
            "Overlays", list(OVERLAY_LABELS), selection_mode="multi", default=["SMA10", "EMA20"],
            format_func=lambda k: OVERLAY_LABELS[k], key="overlays",
        ) or []
    overlays = [o for o in OVERLAY_LABELS if o in overlays]

    view = prices[prices.index > prices.index[-1] - RANGES[rng]]
    if len(view) < 2:
        view = prices.tail(2)
    first, lastp = view["Close"].iloc[0], view["Close"].iloc[-1]
    range_chg = lastp / first - 1
    st.html(
        f'<div class="tp tp-sec-sub" style="margin:4px 0 0">'
        f'{esc(rng)} price change <b class="{direction_class(range_chg)}">{esc(pct(range_chg, 2, signed=True))}</b>'
        f' &nbsp; High {esc(money(view["High"].max(), cur))} &nbsp; Low {esc(money(view["Low"].min(), cur))}'
        f' &nbsp; {len(view):,} sessions</div>'
    )
    top_chart, volume_chart = price_chart(view, overlays, cur)
    with st.container(gap=None):
        st.altair_chart(top_chart, width="stretch")
        if volume_chart is not None:
            st.altair_chart(volume_chart, width="stretch")

with right:
    if res is not None:
        st.html(call_html(res, symbol, live))
    else:
        st.html(f'<div class="tp tp-call"><div class="tp-call-h">Next session</div>'
                f'<p class="tp-verdict-note">{esc(model_error)}</p></div>')

st.write("")
tab_model, tab_ind, tab_data = st.tabs(["Model track record", "Indicators", "Price data"])

with tab_model:
    if res is None:
        st.info(model_error)
    else:
        m = res.metrics
        section(
            "How the model has done on unseen days",
            f"Walk-forward test on {m['test_days']} sessions, {day(m['test_start'])} to {day(m['test_end'])}. "
            f"Retrained {res.backtest['TrainRows'].nunique()} times, each time on data before the block it predicts.",
        )
        st.html(stats_strip([
            ("Accuracy", pct(m["accuracy"]), f"95% range {pct(m['accuracy_ci_low'])} – {pct(m['accuracy_ci_high'])}"),
            ("Baseline", pct(m["baseline"]), f"Calling {m['baseline_label'].split()[-1]} every day"),
            ("Edge", num((m["accuracy"] - m["baseline"]) * 100, 1, signed=True) + " pts", "Accuracy minus baseline"),
            ("ROC AUC", num(m["auc"], 3), "0.500 = no skill"),
            ("Following the model", pct(m["model_return"], 1, signed=True),
             f"Max drawdown {pct(m['model_drawdown'], 1)}"),
            ("Buy and hold", pct(m["buyhold_return"], 1, signed=True),
             f"Max drawdown {pct(m['buyhold_drawdown'], 1)}"),
        ]))

        section("Growth of 100",
                f"Holding {symbol} only on days the model called Up ({pct(m['exposure'], 0)} of days) "
                "versus holding every day. No trading costs or taxes included.")
        st.altair_chart(equity_chart(res.backtest), width="stretch")

        a, b, c = st.columns([1.05, 1, 1.15], gap="large")
        with a:
            section("Calls vs outcomes", "Counts of test sessions")
            st.html(confusion_html(m))
        with b:
            section("Accuracy by test block", "Dashed line marks 50%")
            st.altair_chart(block_chart(res.backtest), width="stretch")
        with c:
            section("What the model leans on", "Share of total split gain, latest fit")
            st.altair_chart(importance_chart(res.importance), width="stretch")

        st.caption(
            f"Final model trained on {res.train_rows:,} sessions from {day(res.train_start)}. "
            "Daily direction is hard to forecast; accuracy in the low-to-mid 50s is typical for this kind of model."
        )

with tab_ind:
    section("RSI (14)", f"Wilder's relative strength index over the {rng} range. Shaded band marks 30 to 70.")
    st.altair_chart(rsi_chart(view), width="stretch")
    section("Model inputs today", "Latest value of each input, compared with this stock's own 10-year history.")
    st.html(inputs_html(data))

with tab_data:
    table = prices[["Open", "High", "Low", "Close", "ChangePct", "Volume", "SMA5", "SMA10", "EMA20", "RSI14"]].copy()
    table["ChangePct"] *= 100
    table = table.iloc[::-1].reset_index()
    section("Daily prices", f"{len(table):,} sessions with indicators, newest first. Prices in {cur or 'local currency'}.")
    st.dataframe(
        table.head(250),
        width="stretch",
        height=420,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "Open": st.column_config.NumberColumn("Open", format="%.2f"),
            "High": st.column_config.NumberColumn("High", format="%.2f"),
            "Low": st.column_config.NumberColumn("Low", format="%.2f"),
            "Close": st.column_config.NumberColumn("Close", format="%.2f"),
            "ChangePct": st.column_config.NumberColumn("Change %", format="%+.2f%%"),
            "Volume": st.column_config.NumberColumn("Volume", format="localized"),
            "SMA5": st.column_config.NumberColumn("SMA 5", format="%.2f"),
            "SMA10": st.column_config.NumberColumn("SMA 10", format="%.2f"),
            "EMA20": st.column_config.NumberColumn("EMA 20", format="%.2f"),
            "RSI14": st.column_config.NumberColumn("RSI 14", format="%.1f"),
        },
    )
    d1, d2, _ = st.columns([1, 1, 2])
    export = prices.join(data[FEATURES])[["Open", "High", "Low", "Close", "Adj Close", "Volume", "ChangePct", "SMA5",
                                          "SMA10", "EMA20", "RSI14", *FEATURES]].round(6)
    with d1:
        st.download_button("Download full history (CSV)", export.to_csv().encode("utf-8"),
                           file_name=f"{symbol}_history.csv", mime="text/csv")
    with d2:
        if res is not None:
            preds = res.backtest[["ProbUp", "Predicted", "Actual", "Correct", "NextReturn"]].round(6)
            preds.loc[res.as_of, ["ProbUp", "Predicted"]] = [round(res.prob_up, 6), int(res.call == "Up")]
            st.download_button("Download predictions (CSV)", preds.to_csv().encode("utf-8"),
                               file_name=f"{symbol}_predictions.csv", mime="text/csv")

st.html(
    '<div class="tp tp-foot">Prices from Yahoo Finance via yfinance. Displayed prices are as quoted '
    "(split-adjusted); the model uses dividend-adjusted prices. Educational project, not investment advice.</div>"
)
