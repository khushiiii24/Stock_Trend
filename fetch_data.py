"""Quick check that Yahoo Finance data loads and looks sane.

Run with:  python fetch_data.py            (defaults to AAPL)
           python fetch_data.py RELIANCE.NS
"""
import sys

from features import load_history

ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
prices, meta = load_history(ticker, "2y")

print(f"{meta['name']} ({meta['symbol']}), {meta['exchange']}, prices in {meta['currency']}")
print(f"{len(prices):,} sessions from {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}")
print(prices.tail().round(2))
