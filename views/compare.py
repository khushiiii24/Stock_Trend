"""Compare: several symbols side by side (performance, risk, correlation)."""
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ui
from features import DataError

LISTS = {
    "US tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"],
    "India IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
    "India banks": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "Indices": ["^GSPC", "^IXIC", "^DJI", "^NSEI", "^BSESN"],
}
MAX_SYMBOLS = len(ui.PALETTE)

if "cmp_syms" not in st.session_state:
    st.session_state.cmp_syms = list(dict.fromkeys([ui.current_symbol(), "MSFT", "NVDA", "GOOGL"]))[:4]


def _apply_list() -> None:
    name = st.session_state.get("cmp_list")
    if name:
        st.session_state.cmp_syms = LISTS[name][:MAX_SYMBOLS]


ui.page_title("Compare stocks", "Line up to six symbols on the same scale: who grew most, who swung most, "
                                "and which ones move together.")

st.pills("Quick lists", list(LISTS), key="cmp_list", on_change=_apply_list)
options = list(dict.fromkeys(sum(LISTS.values(), []) + ui.POPULAR + st.session_state.cmp_syms))
c1, c2 = st.columns([3, 1.3], vertical_alignment="bottom")
with c1:
    chosen = st.multiselect("Symbols", options, key="cmp_syms", max_selections=MAX_SYMBOLS,
                            accept_new_options=True, placeholder="Pick or type symbols")
with c2:
    rng = st.segmented_control("Range", ["1M", "3M", "6M", "1Y", "2Y", "5Y"], default="1Y", key="cmp_range") or "1Y"

chosen = [s.strip().upper() for s in chosen if s and s.strip()]
if not chosen:
    st.info("Pick at least one symbol, or click a quick list above.")
    st.stop()

loaded, failed = {}, []
with st.spinner("Loading prices…"):
    for s in chosen:
        try:
            loaded[s] = ui.load_prices(s)
        except DataError:
            failed.append(s)
if failed:
    st.warning("No data for " + ", ".join(failed) + ". Check the symbols.")
if not loaded:
    st.stop()

colors = {s: ui.PALETTE[i] for i, s in enumerate(loaded)}
offset = ui.RANGES[rng]

# ---- board of today's moves
cells = []
for s, (prices, meta, live) in loaded.items():
    chg = prices["Close"].iloc[-1] / prices["Close"].iloc[-2] - 1
    strength = min(abs(chg) / 0.03, 1)
    alpha = 0.10 + 0.32 * strength
    bg = f"rgba(18,128,79,{alpha:.2f})" if chg > 0 else f"rgba(194,58,47,{alpha:.2f})" if chg < 0 else "#FFFFFF"
    cells.append(
        f'<div class="cell" style="background:{bg}"><div class="s">'
        f'<span class="tp-swatch" style="background:{colors[s]}"></span>{ui.esc(s)}</div>'
        f'<div class="n">{ui.esc(meta["name"])}</div>'
        f'<div class="p">{ui.esc(ui.money(prices["Close"].iloc[-1], meta["currency"]))}</div>'
        f'<div class="c {ui.direction_class(chg)}">{ui.arrow(chg)} {ui.esc(ui.pct(chg, 2, signed=True))}'
        f'{" live" if live else ""}</div></div>'
    )
ui.section("Latest session", "Tile colour gets stronger as the move gets bigger (full colour at ±3%).")
st.html('<div class="tp tp-board">' + "".join(cells) + "</div>")

# ---- performance indexed to 100
indexed, rets, rows = {}, {}, []
for s, (prices, meta, live) in loaded.items():
    view = prices[prices.index > prices.index[-1] - offset]
    if len(view) < 2:
        continue
    indexed[s] = view["Close"] / view["Close"].iloc[0] * 100
    daily = np.log(view["Adj Close"]).diff()
    rets[s] = daily
    year = prices[prices.index > prices.index[-1] - pd.DateOffset(years=1)]
    lo52, hi52 = year["Low"].min(), year["High"].max()
    pos = 0.5 if hi52 == lo52 else (view["Close"].iloc[-1] - lo52) / (hi52 - lo52)
    eq = view["Adj Close"] / view["Adj Close"].iloc[0]
    rows.append({
        "s": s, "name": meta["name"], "cur": meta["currency"], "last": view["Close"].iloc[-1],
        "day": view["Close"].iloc[-1] / prices["Close"].iloc[-2] - 1,
        "ret": view["Close"].iloc[-1] / view["Close"].iloc[0] - 1,
        "vol": daily.std() * math.sqrt(252), "dd": float((eq / eq.cummax() - 1).min()),
        "rsi": prices["RSI14"].iloc[-1], "pos": pos,
    })

perf = pd.DataFrame(indexed).sort_index().ffill()
ui.section(f"Growth of 100 over {rng}", "Each line starts at 100, so stocks in different currencies compare fairly.")
st.altair_chart(ui.lines_chart(perf, list(perf.columns), [colors[s] for s in perf.columns], height=320,
                               baseline=100), width="stretch")

# ---- table
best = max(rows, key=lambda r: r["ret"])["s"] if rows else None
body = []
for r in sorted(rows, key=lambda r: -r["ret"]):
    body.append(
        f'<tr><td><span class="tp-swatch" style="background:{colors[r["s"]]}"></span><b>{ui.esc(r["s"])}</b>'
        f'{" (top)" if r["s"] == best and len(rows) > 1 else ""}</td>'
        f'<td class="l">{ui.esc(r["name"][:34])}</td><td>{ui.esc(ui.money(r["last"], r["cur"]))}</td>'
        f'<td class="{ui.direction_class(r["day"])}">{ui.esc(ui.pct(r["day"], 2, signed=True))}</td>'
        f'<td class="{ui.direction_class(r["ret"])}"><b>{ui.esc(ui.pct(r["ret"], 1, signed=True))}</b></td>'
        f'<td>{ui.esc(ui.pct(r["vol"], 1))}</td><td class="tp-down">{ui.esc(ui.pct(r["dd"], 1))}</td>'
        f'<td>{ui.esc(ui.num(r["rsi"], 1))}</td>'
        f'<td><span class="tp-pos"><i style="left:{r["pos"] * 100:.0f}%"></i></span></td></tr>'
    )
ui.section("Scorecard", f"Sorted by {rng} return. Volatility is annualised; drawdown is the worst fall from a peak.")
st.html(
    '<table class="tp tp-table"><tr><th>Symbol</th><th class="l">Name</th><th>Last</th><th>Day</th>'
    f'<th>{ui.esc(rng)} return</th><th>Volatility</th><th>Max drawdown</th><th>RSI</th><th>52-week position</th></tr>'
    + "".join(body) + "</table>"
)

# ---- correlation
if len(rets) >= 2:
    ret_df = pd.DataFrame(rets).dropna()
    corr = ret_df.corr()
    cdf = corr.reset_index().melt(id_vars="index", var_name="B", value_name="Correlation").rename(columns={"index": "A"})
    order = list(corr.columns)
    heat = alt.Chart(cdf).mark_rect(cornerRadius=4, stroke="#FFFFFF", strokeWidth=2).encode(
        x=alt.X("B:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, orient="top", ticks=False, domain=False)),
        y=alt.Y("A:N", sort=order, title=None, axis=alt.Axis(ticks=False, domain=False)),
        color=alt.Color("Correlation:Q", scale=alt.Scale(domain=[-1, 0, 1], range=["#e34948", "#f0efec", "#2a78d6"],
                                                         interpolate="rgb"),
                        legend=alt.Legend(orient="right", format=".1f", gradientLength=160)),
        tooltip=[alt.Tooltip("A:N", title="Symbol"), alt.Tooltip("B:N", title="With"),
                 alt.Tooltip("Correlation:Q", format=".2f")],
    )
    text = alt.Chart(cdf).mark_text(fontSize=12, fontWeight=500).encode(
        x=alt.X("B:N", sort=order), y=alt.Y("A:N", sort=order), text=alt.Text("Correlation:Q", format=".2f"),
        color=alt.condition("abs(datum.Correlation) > 0.6", alt.value("#FFFFFF"), alt.value("#14213D")))
    size = 64 * len(order)
    ui.section("Do they move together?",
               f"Correlation of daily returns over {rng}, on {len(ret_df):,} sessions all symbols traded. "
               "1 means they move in lockstep, 0 means unrelated.")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.altair_chart(alt.layer(heat, text).properties(width=size, height=size), width=size + 140)
    with c2:
        pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(order) for b in order[i + 1:]]
        hi_pair, lo_pair = max(pairs, key=lambda t: t[2]), min(pairs, key=lambda t: t[2])
        st.html(
            '<div class="tp tp-card"><h4>Reading the grid</h4><ul>'
            f'<li>Closest pair: <b>{ui.esc(hi_pair[0])}</b> and <b>{ui.esc(hi_pair[1])}</b> at {hi_pair[2]:.2f}.</li>'
            f'<li>Most independent pair: <b>{ui.esc(lo_pair[0])}</b> and <b>{ui.esc(lo_pair[1])}</b> at {lo_pair[2]:.2f}.</li>'
            "<li>Holding stocks with lower correlation spreads risk: they are less likely to fall on the same day.</li>"
            "</ul></div>"
        )

ui.footer()
