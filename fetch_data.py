import yfinance as yf

data = yf.download("AAPL", period="2y")
print(data.head())