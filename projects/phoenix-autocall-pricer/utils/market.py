import yfinance as yf
import numpy as np


def get_market_data(ticker="^STOXX50E", period="2y"):
    data = yf.download(ticker, period=period, auto_adjust=True)

    prices = data["Close"].squeeze()  # Ensure 1-D Series even for single-ticker downloads

    returns = np.log(prices / prices.shift(1)).dropna()

    S0 = float(prices.iloc[-1])  # Cast to plain Python float to avoid downstream shape issues

    # Annualized volatility
    sigma = float(returns.std() * np.sqrt(252))

    return S0, sigma, prices