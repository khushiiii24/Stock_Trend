"""Guess the move: pick Higher or Lower for a real past session and see if you beat the model."""
import numpy as np
import pandas as pd
import streamlit as st

import ui

ui.page_title("Guess the move", "You see a real chart up to a random day from the last year. Guess whether the "
                                "next session closed higher or lower, then see whether you beat the model.")
symbol = ui.symbol_picker()
prices, meta, data, live, res, model_error = ui.load_model_or_stop(symbol)
if res is None:
    st.info(model_error)
    st.stop()
bt = res.backtest
cur = meta["currency"]
state = st.session_state
if "score" not in state:
    state.score = {"rounds": 0, "you": 0, "model": 0, "streak": 0, "best": 0}
if "played" not in state:
    state.played = set()


def new_round(sym: str) -> None:
    choices = [d for d in bt.index if (sym, d) not in state.played]
    if not choices:
        state.played = {k for k in state.played if k[0] != sym}
        choices = list(bt.index)
    pick = choices[np.random.default_rng().integers(len(choices))]
    state.game = {"symbol": sym, "date": pick, "guess": None}


def make_guess(direction: str) -> None:
    g = state.game
    if g["guess"] is not None:
        return
    row = bt.loc[g["date"]]
    you_right = (direction == "Up") == (row["Actual"] == 1)
    model_right = bool(row["Correct"])
    g.update(guess=direction, you_right=you_right, model_right=model_right)
    sc = state.score
    sc["rounds"] += 1
    sc["you"] += int(you_right)
    sc["model"] += int(model_right)
    sc["streak"] = sc["streak"] + 1 if you_right else 0
    sc["best"] = max(sc["best"], sc["streak"])
    state.played.add((g["symbol"], g["date"]))


def reset_score() -> None:
    state.score = {"rounds": 0, "you": 0, "model": 0, "streak": 0, "best": 0}


if "game" not in state or state.game["symbol"] != symbol or state.game["date"] not in bt.index:
    new_round(symbol)
g = state.game
d = g["date"]
sc = state.score

lead = "even" if sc["you"] == sc["model"] else ("ahead" if sc["you"] > sc["model"] else "behind")
st.html(ui.tiles_html([
    ("Rounds played", f"{sc['rounds']}", "Across all symbols this visit", ""),
    ("Your hits", f"{sc['you']} of {sc['rounds']}", ui.pct(sc["you"] / sc["rounds"], 0) + " right" if sc["rounds"] else "No guesses yet",
     "up" if lead == "ahead" else ""),
    ("Model hits", f"{sc['model']} of {sc['rounds']}",
     ui.pct(sc["model"] / sc["rounds"], 0) + " right" if sc["rounds"] else "Same days as you", "down" if lead == "behind" else ""),
    ("Streak", f"{sc['streak']}", f"Best so far {sc['best']}", "cobalt" if sc["streak"] >= 3 else ""),
], cols=4))

pos = prices.index.get_loc(d)
revealed = g["guess"] is not None
window = prices.iloc[max(0, pos - 59): pos + (2 if revealed else 1)]
row = prices.loc[d]
ema_gap = row["Close"] / row["EMA20"] - 1
ret5 = row["Close"] / prices["Close"].iloc[max(0, pos - 5)] - 1

left, right = st.columns([2, 1], gap="large")
with left:
    ui.section(f"{symbol}, 60 sessions up to {ui.day(d, weekday=True)}",
               "The dashed line marks the day you're guessing from." if revealed else
               "What happens after this day is hidden until you guess.")
    ui.show_price_chart(window, ["EMA20"], cur, height=300, marker=d if revealed else None)
with right:
    ui.section("Clues on that day")
    rsi = row["RSI14"]
    st.html(ui.tiles_html([
        ("Close", ui.money(row["Close"], cur), f"Day change {ui.pct(row['ChangePct'], 2, signed=True)}",
         ui.tone(row["ChangePct"])),
        ("5-day move", ui.pct(ret5, 2, signed=True), "Close vs 5 sessions earlier", ui.tone(ret5)),
        ("RSI (14)", ui.num(rsi, 1), "Overbought" if rsi >= 70 else "Oversold" if rsi <= 30 else "Neutral",
         "amber" if rsi >= 70 else "cobalt" if rsi <= 30 else ""),
        ("vs 20-day EMA", ui.pct(ema_gap, 2, signed=True), "Above the trend line" if ema_gap > 0 else "Below the trend line",
         ""),
    ], cols=2))

    if not revealed:
        st.html('<div class="tp tp-sec" style="margin-top:4px">Next session closes…</div>')
        b1, b2 = st.columns(2)
        with b1:
            st.button("▲ Higher", key="guess_up", on_click=make_guess, args=("Up",), width="stretch")
        with b2:
            st.button("▼ Lower", key="guess_down", on_click=make_guess, args=("Down",), width="stretch")
    else:
        b = bt.loc[d]
        move = b["NextReturn"]
        went = "higher" if b["Actual"] == 1 else "lower"
        model_call = "Higher" if b["Predicted"] == 1 else "Lower"
        title = "You got it!" if g["you_right"] else "Not this time"
        st.html(
            f'<div class="tp tp-reveal {"win" if g["you_right"] else "lose"}"><h3>{title}</h3>'
            f'<p>The next session closed <b>{went}</b>, a move of '
            f'<b class="{ui.direction_class(move)}">{ui.esc(ui.pct(move, 2, signed=True))}</b>.</p>'
            f'<p>You said <b>{"Higher" if g["guess"] == "Up" else "Lower"}</b>. The model said <b>{model_call}</b> '
            f'with a {ui.esc(ui.pct(b["ProbUp"], 1))} chance of a higher close, and was '
            f'<b>{"right" if g["model_right"] else "wrong"}</b>.</p></div>'
        )
        st.button("Next round", type="primary", on_click=new_round, args=(symbol,), width="stretch",
                  icon=":material/casino:")
    st.button("Reset score", type="tertiary", on_click=reset_score, icon=":material/restart_alt:")

st.caption("Days come from the model's walk-forward test year, so its calls here are genuine out-of-sample predictions.")
ui.footer()
