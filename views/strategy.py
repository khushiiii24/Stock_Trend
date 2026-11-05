"""Strategy lab: turn the model's out-of-sample calls into a trading rule and see what it would have done."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ui

ui.page_title("Strategy lab", "Turn the model's calls into a simple trading rule and see how it would have done "
                              "over the last year of days the model had never seen.")
symbol = ui.symbol_picker()
prices, meta, data, live, res, model_error = ui.load_model_or_stop(symbol)
if res is None:
    st.info(model_error)
    st.stop()
cur = meta["currency"]

with st.container(horizontal=True, gap="large", vertical_alignment="bottom"):
    capital = st.number_input(f"Starting amount ({cur})", min_value=100.0, max_value=1e9, value=10_000.0,
                              step=1_000.0, format="%.0f", key="lab_capital", width=200)
    threshold = st.slider("Buy when chance of a higher close is at least", 50, 70, 52, step=1, format="%d%%",
                          key="lab_threshold", width=330)
    cost = st.slider("Cost per trade", 0.0, 0.5, 0.1, step=0.05, format="%.2f%%", key="lab_cost", width=230,
                     help="Brokerage, taxes and slippage, charged each time you buy or sell.")
    mode = st.segmented_control("When the model says Down", ["Stay in cash", "Go short"], default="Stay in cash",
                                key="lab_mode") or "Stay in cash"

bt = res.backtest
p, r = bt["ProbUp"].to_numpy(), bt["NextReturn"].to_numpy()
th = threshold / 100
pos = np.where(p >= th, 1, np.where((mode == "Go short") & (p <= 1 - th), -1, 0))
turnover = np.abs(np.diff(np.concatenate([[0], pos])))
daily = pos * r - turnover * cost / 100
equity = capital * np.cumprod(1 + daily)
buyhold = capital * np.cumprod(1 + r)
curve = pd.DataFrame({"Your rule": equity, "Buy and hold": buyhold, "Position": pos, "P(up)": p,
                      "Next day": r}, index=bt.index)
curve.index.name = "Date"

# trades = runs of the same non-zero position
trades, start = [], None
for i, (d, v) in enumerate(zip(bt.index, pos)):
    if v != 0 and (i == 0 or pos[i - 1] != v):
        start = i
    if v != 0 and (i == len(pos) - 1 or pos[i + 1] != v):
        seg = slice(start, i + 1)
        gross = float(np.prod(1 + v * r[seg]) - 1)
        exit_day = bt.index[i + 1] if i + 1 < len(bt.index) else bt.index[i] + pd.offsets.BDay(1)
        trades.append({"Direction": "Long" if v > 0 else "Short", "Entry": bt.index[start], "Exit": exit_day, "Sessions": i - start + 1,
                       "Return %": round(gross * 100, 2)})
trade_df = pd.DataFrame(trades)

in_market = pos != 0
wins = (pos * r > 0)[in_market]
final = equity[-1]
total = final / capital - 1
bh_total = buyhold[-1] / capital - 1
dd = float((pd.Series(equity) / pd.Series(equity).cummax() - 1).min())
bh_dd = float((pd.Series(buyhold) / pd.Series(buyhold).cummax() - 1).min())
costs_paid = float((turnover * cost / 100 * np.concatenate([[capital], equity[:-1]])).sum())

edge_pts = round((total - bh_total) * 100, 1)
st.html(ui.tiles_html([
    ("Final value", ui.money(final, cur, 0), f"{ui.pct(total, 1, signed=True)} on {ui.money(capital, cur, 0)}",
     ui.tone(total)),
    ("Vs buy and hold", ui.num(edge_pts, 1, signed=True) + " pts", f"Buy and hold {ui.pct(bh_total, 1, signed=True)}",
     ui.tone(edge_pts)),
    ("Trades", f"{len(trade_df):,}", f"Costs paid {ui.money(costs_paid, cur, 0)}", ""),
    ("Winning days", ui.pct(wins.mean(), 1) if in_market.any() else "n/a",
     f"Of {int(in_market.sum())} days in the market", ""),
    ("Max drawdown", ui.pct(dd, 1), f"Buy and hold {ui.pct(bh_dd, 1)}", "down" if dd < bh_dd else ""),
    ("Time in market", ui.pct(in_market.mean(), 0), f"{ui.day(bt.index[0])} to {ui.day(bt.index[-1])}", ""),
]))

if edge_pts > 0:
    verdict = (f"Your rule finished {ui.num(edge_pts, 1)} percentage points ahead of simply holding {symbol}. "
               "One year is a short test, so try other thresholds and symbols before trusting it.")
    cls = "up"
elif edge_pts < 0:
    verdict = (f"Your rule finished {ui.num(-edge_pts, 1)} percentage points behind simply holding {symbol}. "
               "Trading costs and missed up-days usually explain the gap.")
    cls = "down"
else:
    verdict = f"Your rule finished level with simply holding {symbol}."
    cls = ""
st.html(f'<div class="tp tp-reading tp-tile {cls}" style="margin:-6px 0 18px">{ui.esc(verdict)}</div>')

shade = []
if len(trade_df):
    for t in trades:
        shade.append({"start": t["Entry"], "end": t["Exit"], "kind": t["Direction"]})
ui.section(f"Value of {ui.money(capital, cur, 0)} over the test year",
           "Shaded bands show when your rule was holding (green) or short (red). Hover for daily detail.")
st.altair_chart(
    ui.lines_chart(curve, ["Your rule", "Buy and hold"], [ui.COBALT, ui.BENCHMARK], height=320, value_fmt=",.0f",
                   baseline=capital, shading=pd.DataFrame(shade),
                   tooltip_extra=[alt.Tooltip("P(up):Q", format=".1%"),
                                  alt.Tooltip("Next day:Q", title="Next-day move", format="+.2%")]),
    width="stretch",
)

left, right = st.columns([1.4, 1], gap="large")
with left:
    ui.section("Trade log", "Newest first. Returns are before costs.")
    if len(trade_df):
        show = trade_df[["Direction", "Entry", "Exit", "Sessions", "Return %"]].iloc[::-1]
        st.dataframe(show, hide_index=True, width="stretch", height=300, column_config={
            "Entry": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Exit": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Return %": st.column_config.NumberColumn(format="%+.2f%%"),
        })
    else:
        st.info("No trades at this threshold. Lower it to let the rule buy.")
with right:
    ui.section("How the rule works")
    st.html(
        '<div class="tp tp-card"><ul>'
        f"<li>Each evening the model gives a chance that {ui.esc(symbol)} closes higher the next day.</li>"
        f"<li>If that chance is at least <b>{threshold}%</b>, you hold the stock for the next session.</li>"
        + ("<li>If the chance of a higher close is at most "
           f"<b>{100 - threshold}%</b>, you go short instead.</li>" if mode == "Go short" else
           "<li>Otherwise you stay in cash and earn nothing that day.</li>")
        + f"<li>Every buy or sell costs <b>{cost:.2f}%</b> of the amount traded.</li>"
        "<li>All calls come from the walk-forward test, so the model never saw the day it was predicting.</li>"
        "</ul></div>"
    )

ui.footer()
