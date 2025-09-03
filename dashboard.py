import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from features import add_features
import joblib
import pandas as pd
from io import BytesIO

# Load trained model
model = joblib.load("xgb_model.pkl")
features = ["Close", "Return", "SMA5", "SMA10", "EMA20", "Volatility"]

# Streamlit App Config
st.set_page_config(page_title="Stock Market Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Market Trend Predictor")
st.markdown("Enter a stock ticker symbol and get the **predicted trend** for the next trading day.")

# User Inputs
col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
with col2:
    period = st.selectbox("Select Period", ["6mo", "1y", "2y", "5y", "10y"], index=1)  # ✅ Added 10y option

# Fetch data
with st.spinner("Fetching stock data..."):
    df = yf.download(ticker, period=period)
    df = add_features(df)

# Check if data is available
if df.empty:
    st.error("No data found. Please try another ticker.")
else:
    # Predict next-day trend
    latest = df[features].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(latest)[0]

    # Display prediction result
    if prediction == 1:
        st.success(f"📈 Prediction: **UP** for next trading day")
    else:
        st.error(f"📉 Prediction: **DOWN** for next trading day")

    # Plot price history + SMA lines
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Close"], label="Close Price", color="blue")
    if "SMA5" in df.columns:
        ax.plot(df.index, df["SMA5"], label="SMA5", linestyle="--", color="green")
    if "SMA10" in df.columns:
        ax.plot(df.index, df["SMA10"], label="SMA10", linestyle="--", color="orange")
    ax.set_title(f"{ticker} Price History ({period})", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # Show last few rows as table
    st.subheader(f"📊 Recent Data for {ticker}")
    last_10 = df.tail(10)
    st.dataframe(last_10)

    # Option: Show previous 1 year data separately
    if st.checkbox("Show Previous 1 Year Data"):
        st.write("📅 Previous 1 Year Historical Data")
        df_1y = yf.download(ticker, period="1y")
        df_1y = add_features(df_1y)
        st.dataframe(df_1y.tail(20))

    # Prepare CSV data
    output = BytesIO()
    csv_data = last_10.copy()
    csv_data["Prediction"] = "UP" if prediction == 1 else "DOWN"
    csv_data.to_csv(output, index=True)
    output.seek(0)

    # Download button
    st.download_button(
        label="📥 Download Last 10 Days + Prediction as CSV",
        data=output,
        file_name=f"{ticker}_prediction.csv",
        mime="text/csv",
    )
