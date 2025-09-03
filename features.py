import pandas as pd

def add_features(df):
    df["Return"] = df["Close"].pct_change()
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Volatility"] = df["Return"].rolling(10).std()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna()
