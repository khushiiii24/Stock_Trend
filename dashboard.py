"""Stock Trend Predictor: app entry point.

Run with:  streamlit run dashboard.py
Each page lives in views/; shared styling, data loading and charts live in ui.py.
"""
import streamlit as st

st.set_page_config(
    page_title="Stock Trend Predictor",
    page_icon=":material/show_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from ui import inject_css  # noqa: E402  (page config must run first)

inject_css()

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/monitoring:", default=True),
    st.Page("views/compare.py", title="Compare", icon=":material/stacked_line_chart:", url_path="compare"),
    st.Page("views/strategy.py", title="Strategy lab", icon=":material/science:", url_path="strategy"),
    st.Page("views/game.py", title="Guess the move", icon=":material/sports_esports:", url_path="game"),
    st.Page("views/learn.py", title="Learn", icon=":material/school:", url_path="learn"),
]
st.navigation(pages, position="top").run()
