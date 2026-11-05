"""Overview: quote, chart, next-session forecast and the model's track record."""
import pandas as pd
import streamlit as st

import ui
from features import FEATURES

symbol = ui.symbol_picker()
prices, meta, data, live, res, model_error = ui.load_model_or_stop(symbol)
cur = meta["currency"]

st.html(ui.hero_html(meta, prices, live))
st.html(ui.quote_tiles(meta, prices, live))

left, right = st.columns([2.1, 1], gap="large")
with left:
    with st.container(horizontal=True, gap="medium", vertical_alignment="bottom"):
        rng = st.segmented_control("Range", list(ui.RANGES), default="1Y", key="range") or "1Y"
        overlays = st.segmented_control(
            "Overlays", list(ui.OVERLAY_LABELS), selection_mode="multi", default=["SMA10", "EMA20"],
            format_func=lambda k: ui.OVERLAY_LABELS[k], key="overlays",
        ) or []
    overlays = [o for o in ui.OVERLAY_LABELS if o in overlays]

    view = prices[prices.index > prices.index[-1] - ui.RANGES[rng]]
    if len(view) < 2:
        view = prices.tail(2)
    range_chg = view["Close"].iloc[-1] / view["Close"].iloc[0] - 1
    st.html(
        f'<div class="tp tp-sec-sub" style="margin:4px 0 0">'
        f'{ui.esc(rng)} price change <b class="{ui.direction_class(range_chg)}">{ui.esc(ui.pct(range_chg, 2, signed=True))}</b>'
        f' &nbsp; High {ui.esc(ui.money(view["High"].max(), cur))} &nbsp; Low {ui.esc(ui.money(view["Low"].min(), cur))}'
        f' &nbsp; {len(view):,} sessions</div>'
    )
    ui.show_price_chart(view, overlays, cur)

with right:
    if res is not None:
        st.html(ui.call_html(res, symbol, live))
        c1, c2 = st.columns(2)
        with c1:
            st.page_link("views/strategy.py", label="Test a rule", icon=":material/science:")
        with c2:
            st.page_link("views/game.py", label="Beat the model", icon=":material/sports_esports:")
    else:
        st.html(f'<div class="tp tp-call"><div class="tp-call-h">Next session</div>'
                f'<p class="tp-note plain">{ui.esc(model_error)}</p></div>')

st.write("")
tab_model, tab_ind, tab_data = st.tabs(["Model track record", "Indicators", "Price data"])

with tab_model:
    if res is None:
        st.info(model_error)
    else:
        m = res.metrics
        ui.section(
            "How the model has done on unseen days",
            f"Walk-forward test on {m['test_days']} sessions, {ui.day(m['test_start'])} to {ui.day(m['test_end'])}. "
            f"Retrained {res.backtest['TrainRows'].nunique()} times, each time on data before the block it predicts.",
        )
        edge = round((m["accuracy"] - m["baseline"]) * 100, 1)
        st.html(ui.tiles_html([
            ("Accuracy", ui.pct(m["accuracy"]), f"95% range {ui.pct(m['accuracy_ci_low'])} – {ui.pct(m['accuracy_ci_high'])}", ""),
            ("Baseline", ui.pct(m["baseline"]), f"Calling {m['baseline_label'].split()[-1]} every day", ""),
            ("Edge", ui.num(edge, 1, signed=True) + " pts", "Accuracy minus baseline", ui.tone(edge)),
            ("ROC AUC", ui.num(m["auc"], 3), "0.500 = no skill", ""),
            ("Following the model", ui.pct(m["model_return"], 1, signed=True),
             f"Max drawdown {ui.pct(m['model_drawdown'], 1)}", ui.tone(m["model_return"])),
            ("Buy and hold", ui.pct(m["buyhold_return"], 1, signed=True),
             f"Max drawdown {ui.pct(m['buyhold_drawdown'], 1)}", ui.tone(m["buyhold_return"])),
        ]))
        ui.section("Growth of 100",
                   f"Holding {symbol} only on days the model called Up ({ui.pct(m['exposure'], 0)} of days) "
                   "versus holding every day. No trading costs or taxes included.")
        eq = res.backtest[["ModelEquity", "BuyHoldEquity"]].rename(
            columns={"ModelEquity": "Following the model", "BuyHoldEquity": "Buy and hold"})
        st.altair_chart(ui.lines_chart(eq, list(eq.columns), [ui.COBALT, ui.BENCHMARK], baseline=100),
                        width="stretch")

        a, b, c = st.columns([1.05, 1, 1.15], gap="large")
        with a:
            ui.section("Calls vs outcomes", "Counts of test sessions")
            st.html(ui.confusion_html(m))
        with b:
            ui.section("Accuracy by test block", "Dashed line marks 50%")
            st.altair_chart(ui.block_chart(res.backtest), width="stretch")
        with c:
            ui.section("What the model leans on", "Share of total split gain, latest fit")
            st.altair_chart(ui.importance_chart(res.importance), width="stretch")
        st.caption(
            f"Final model trained on {res.train_rows:,} sessions from {ui.day(res.train_start)}. "
            "Daily direction is hard to forecast; accuracy in the low-to-mid 50s is typical for this kind of model."
        )

with tab_ind:
    ui.section("RSI (14)", f"Wilder's relative strength index over the {rng} range.")
    st.altair_chart(ui.rsi_chart(view), width="stretch")
    ui.section("Model inputs today", "Latest value of each input, compared with this stock's own 10-year history.")
    st.html(ui.inputs_html(data))
    st.page_link("views/learn.py", label="What do these indicators mean?", icon=":material/school:")

with tab_data:
    table = prices[["Open", "High", "Low", "Close", "ChangePct", "Volume", "SMA5", "SMA10", "EMA20", "RSI14"]].copy()
    table["ChangePct"] *= 100
    table = table.iloc[::-1].reset_index()
    ui.section("Daily prices", f"{len(table):,} sessions with indicators, newest first. Prices in {cur or 'local currency'}.")
    st.dataframe(
        table.head(250), width="stretch", height=420, hide_index=True,
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
    export = prices.join(data[FEATURES])[["Open", "High", "Low", "Close", "Adj Close", "Volume", "ChangePct", "SMA5",
                                          "SMA10", "EMA20", "RSI14", *FEATURES]].round(6)
    with st.container(horizontal=True):
        st.download_button("Download full history (CSV)", export.to_csv().encode("utf-8"),
                           file_name=f"{symbol}_history.csv", mime="text/csv", icon=":material/download:")
        if res is not None:
            preds = res.backtest[["ProbUp", "Predicted", "Actual", "Correct", "NextReturn"]].round(6)
            preds.loc[res.as_of, ["ProbUp", "Predicted"]] = [round(res.prob_up, 6), int(res.call == "Up")]
            st.download_button("Download predictions (CSV)", preds.to_csv().encode("utf-8"),
                               file_name=f"{symbol}_predictions.csv", mime="text/csv", icon=":material/download:")

ui.footer()
