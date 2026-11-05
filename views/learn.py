"""Learn: what each indicator means, read live on the chosen symbol, plus a short quiz."""
import altair as alt
import pandas as pd
import streamlit as st

import ui

ui.page_title("Learn the indicators", "What the model looks at, explained in plain words and shown on a real chart. "
                                      "Finish with a four-question quiz.")
symbol = ui.symbol_picker()
prices, meta, live = ui.load_prices_or_stop(symbol)
cur = meta["currency"]

TOPICS = ["RSI", "Moving averages", "MACD", "Bollinger bands", "How the model is tested"]
topic = st.segmented_control("Topic", TOPICS, default="RSI", key="learn_topic") or "RSI"
view = prices[prices.index > prices.index[-1] - pd.DateOffset(months=6)].copy()
last = prices.iloc[-1]


def explain(title: str, body: str, points: list[str]) -> None:
    st.html(f'<div class="tp tp-card"><h4>{ui.esc(title)}</h4><p style="margin:0">{body}</p><ul>'
            + "".join(f"<li>{p}</li>" for p in points) + "</ul></div>")


def reading(text: str, cls: str = "") -> None:
    st.html(f'<div class="tp tp-reading tp-tile {cls}" style="margin-top:10px"><b>Today on {ui.esc(symbol)}:</b> '
            f"{text}</div>")


left, right = st.columns([1.9, 1], gap="large")

if topic == "RSI":
    with left:
        ui.section("RSI (14) over the last 6 months")
        st.altair_chart(ui.rsi_chart(view, height=300), width="stretch")
    with right:
        explain("Relative strength index",
                "RSI compares the size of recent gains with recent losses on a 0 to 100 scale.",
                ["Above 70 is called <b>overbought</b>: the stock has risen fast and may pause.",
                 "Below 30 is called <b>oversold</b>: it has fallen fast and may bounce.",
                 "Between 30 and 70 is the normal zone, where RSI says little on its own."])
        r = last["RSI14"]
        zone = "overbought" if r >= 70 else "oversold" if r <= 30 else "in the neutral zone"
        reading(f"RSI is <b>{r:.1f}</b>, {zone}.", "amber" if r >= 70 else "cobalt" if r <= 30 else "")

elif topic == "Moving averages":
    with left:
        ui.section("Price with 5-day, 10-day and 20-day averages")
        ui.show_price_chart(view, ["SMA5", "SMA10", "EMA20"], cur, height=300)
    with right:
        explain("Moving averages",
                "A moving average smooths daily noise by averaging recent closes. Short ones react fast, long ones slowly.",
                ["<b>SMA</b> weights every day equally; <b>EMA</b> gives recent days more weight.",
                 "Price above its average suggests an uptrend; below suggests a downtrend.",
                 "When a short average crosses above a longer one, traders call it a bullish crossover."])
        gap = last["Close"] / last["EMA20"] - 1
        cross = "above" if last["SMA5"] > last["SMA10"] else "below"
        reading(f"Close is <b>{ui.esc(ui.pct(gap, 2, signed=True))}</b> versus its 20-day EMA, and the 5-day "
                f"average is {cross} the 10-day.", ui.tone(gap))

elif topic == "MACD":
    close = prices["Close"]
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    m = pd.DataFrame({"MACD": macd, "Signal": signal, "Histogram": macd - signal}).loc[view.index]
    m.index.name = "Date"
    df = m.reset_index()
    df["Side"] = df["Histogram"].apply(lambda v: "Above signal" if v >= 0 else "Below signal")
    x = alt.X("Date:T", axis=ui.date_axis(180))
    bars = alt.Chart(df).mark_bar(size=3, opacity=0.7).encode(
        x=x, y=alt.Y("Histogram:Q", title=None, axis=alt.Axis(orient="right")),
        color=alt.Color("Side:N", scale=alt.Scale(domain=["Above signal", "Below signal"], range=[ui.UP, ui.DOWN]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"), alt.Tooltip("MACD:Q", format=".2f"),
                 alt.Tooltip("Signal:Q", format=".2f"), alt.Tooltip("Histogram:Q", format="+.2f")])
    long = df.melt(id_vars="Date", value_vars=["MACD", "Signal"], var_name="Line", value_name="Value")
    lines = alt.Chart(long).mark_line(strokeWidth=1.8).encode(
        x=x, y="Value:Q", color=alt.Color("Line:N", scale=alt.Scale(domain=["MACD", "Signal"],
                                                                     range=[ui.COBALT, "#eb6834"]), legend=None))
    with left:
        ui.section("MACD over the last 6 months", "Blue: MACD line. Orange: signal line. Bars: the gap between them.")
        st.altair_chart(alt.layer(bars, lines).resolve_scale(color="independent").properties(height=300),
                        width="stretch")
    with right:
        explain("Moving average convergence divergence",
                "MACD is the 12-day EMA minus the 26-day EMA. Its 9-day average is the signal line.",
                ["Green bars: MACD is above its signal, so momentum is building.",
                 "Red bars: MACD is below its signal, so momentum is fading.",
                 "The model uses the bar height divided by price, so it compares across stocks."])
        h = m["Histogram"].iloc[-1]
        side = m["Histogram"] >= 0
        days_since = int((side[::-1] == side.iloc[-1]).cumprod().sum())
        reading(f"MACD is <b>{'above' if h >= 0 else 'below'}</b> its signal line by {abs(h):.2f}, "
                f"for the last {days_since} session{'s' if days_since != 1 else ''}.", "up" if h >= 0 else "down")

elif topic == "Bollinger bands":
    mid = prices["Close"].rolling(20).mean()
    sd = prices["Close"].rolling(20).std()
    b = pd.DataFrame({"Close": prices["Close"], "Middle": mid, "Upper": mid + 2 * sd, "Lower": mid - 2 * sd}).loc[view.index]
    b.index.name = "Date"
    df = b.reset_index()
    x = alt.X("Date:T", axis=ui.date_axis(180))
    band = alt.Chart(df).mark_area(color=ui.COBALT, opacity=0.12).encode(
        x=x, y=alt.Y("Lower:Q", scale=alt.Scale(zero=False), title=None, axis=alt.Axis(orient="right")), y2="Upper:Q")
    midl = alt.Chart(df).mark_line(color=ui.INK_RULE, strokeDash=[4, 3], strokeWidth=1).encode(x=x, y="Middle:Q")
    closel = alt.Chart(df).mark_line(color=ui.SERIES["Close"], strokeWidth=2).encode(
        x=x, y="Close:Q", tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"), alt.Tooltip("Close:Q", format=",.2f"),
                                   alt.Tooltip("Upper:Q", format=",.2f"), alt.Tooltip("Lower:Q", format=",.2f")])
    with left:
        ui.section("Bollinger bands over the last 6 months", "Shaded band: 20-day average ± 2 standard deviations.")
        st.altair_chart(alt.layer(band, midl, closel).properties(height=300), width="stretch")
    with right:
        explain("Bollinger bands",
                "The bands widen when prices swing a lot and narrow when they are calm.",
                ["Price near the <b>upper</b> band means it is high compared with the last 20 days.",
                 "Price near the <b>lower</b> band means it is low compared with the last 20 days.",
                 "The model uses the position inside the band: 0 is the lower band, 1 the upper."])
        row = b.iloc[-1]
        posb = (row["Close"] - row["Lower"]) / (row["Upper"] - row["Lower"])
        if posb > 1:
            where = "above the upper band"
        elif posb < 0:
            where = "below the lower band"
        else:
            where = "near the upper band" if posb > 0.8 else "near the lower band" if posb < 0.2 else "inside the band"
        reading(f"Close sits at <b>{ui.esc(ui.num(posb, 2))}</b> of the band width, {where}. Band width is "
                f"{ui.esc(ui.pct((row['Upper'] - row['Lower']) / row['Middle'], 1))} of price.",
                "amber" if posb > 0.8 else "cobalt" if posb < 0.2 else "")

else:
    with left:
        ui.section("Walk-forward testing", "How accuracy is measured without letting the model peek at the answers.")
        rows = []
        for i in range(5):
            train = 75 + i * 5
            rows.append(f'<div class="r"><span>Round {i + 1}</span><div class="bar">'
                        f'<div class="tr" style="width:{train}%"></div><div class="te" style="width:5%"></div></div></div>')
        st.html('<div class="tp tp-card"><div class="tp-wf">' + "".join(rows) + "</div>"
                '<div class="tp-legend"><span><span class="tp-swatch" style="background:#3355D6"></span>Training data</span>'
                '<span><span class="tp-swatch" style="background:#eda100"></span>Test block (never seen)</span></div></div>')
    with right:
        explain("Why test this way",
                "Markets change over time, so the test must only ever predict the future from the past.",
                ["The last year is split into five blocks of about 50 sessions.",
                 "For each block the model is retrained on everything before it, then predicts the block.",
                 "Accuracy is compared with a baseline that always guesses the more common direction.",
                 "The 95% range shows how much the accuracy could move by luck alone."])

st.write("")
ui.section("Quick quiz", "Four questions. Pick an answer for each, then check.")
QUIZ = [
    ("RSI is 78. What do traders usually call that?", ["Oversold", "Overbought", "Neutral"], 1,
     "Above 70 is the overbought zone."),
    ("Which average reacts faster to new prices?", ["20-day SMA", "20-day EMA", "They react the same"], 1,
     "An EMA gives recent days more weight, so it turns sooner."),
    ("A model is right 53% of the time, and the stock rose on 54% of days. Is it useful?",
     ["Yes, 53% beats a coin flip", "Not really, it trails simply guessing Up"], 1,
     "Always guessing Up would have scored 54%, so the model adds nothing."),
    ("Why does the walk-forward test retrain before each block?",
     ["To use more data and stay honest about the future", "To make the accuracy look higher"], 0,
     "Each block is predicted only from earlier data, as it would be in real life."),
]
answers = []
cols = st.columns(2, gap="large")
for i, (q, opts, correct, why) in enumerate(QUIZ):
    with cols[i % 2]:
        answers.append(st.radio(f"{i + 1}. {q}", opts, index=None, key=f"quiz_{i}"))
if st.button("Check answers", type="primary", icon=":material/fact_check:"):
    score = sum(1 for (q, opts, correct, why), a in zip(QUIZ, answers) if a == opts[correct])
    cls = "up" if score == len(QUIZ) else "amber" if score >= 2 else "down"
    lines = "".join(
        f"<li>{'✓' if a == opts[c] else '✗'} Q{i + 1}: {ui.esc(why)}</li>"
        for i, ((q, opts, c, why), a) in enumerate(zip(QUIZ, answers))
    )
    st.html(f'<div class="tp tp-reading tp-tile {cls}"><b>You scored {score} out of {len(QUIZ)}.</b>'
            f'<ul style="margin:6px 0 0 18px;padding:0">{lines}</ul></div>')

ui.footer()
