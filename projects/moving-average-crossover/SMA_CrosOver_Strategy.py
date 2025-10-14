# Don't forget to install requirements.text

import yfinance as yf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# Download stock data
ticker = "AAPL"
period = "5y"
interval = "1d"
raw = yf.download(ticker, period=period, interval=interval, auto_adjust = False)

# Explore data
df = raw[['Adj Close']].rename(columns={'Adj Close': 'price'})
df = df.sort_index().dropna()
print(df.head())
df.info()
df.isna().sum()
df.shape

# Compute moving averages
fast_window = 20
slow_window = 50
df['sma_fast'] = df['price'].rolling(window=fast_window, min_periods=fast_window).mean()
df['sma_slow'] = df['price'].rolling(window=slow_window, min_periods=slow_window).mean()
df.head(60)

# Plot moving averages 
plt.figure(figsize=(12,6))
plt.plot(df['price'], label='Price')
plt.plot(df['sma_fast'], label=f'SMA {fast_window}', color='orange', linewidth=2)
plt.plot(df['sma_slow'], label=f'SMA {slow_window}', color='green', linewidth=2, alpha=0.8)
plt.legend()
plt.title('AAPL with 20-day & 50-day SMAs')
plt.show()

# Define trading Strategy
df['signal_raw'] = (df['sma_fast'] > df['sma_slow']).astype(int)
df['signal_confirm'] = (df['signal_raw'].rolling(2).sum() == 2).astype(int)
df['signal'] = df['signal_confirm'].shift(1).fillna(0)
df['trade'] = df['signal'].diff().fillna(0)
entries = (df['trade'] == 1).sum()
exits   = (df['trade'] == -1).sum()
entries, exits

# Backtest the trading strategy 

#1) Daily returns 
df['daily_ret'] = df['price'].pct_change().fillna(0)

# 2) Strategy gross return
df['strat_ret_gross'] = df['signal'] * df['daily_ret']

# 3) Simple trading cost model
cost_per_switch = 0.001
df['cost'] = (df['trade'].abs() > 0).astype(int) * cost_per_switch

# 4) Net strategy return after costs
df['strat_ret_net'] = df['strat_ret_gross'] - df['cost']

# 5) Cumulative growth of $1
df['bh_cum']     = (1 + df['daily_ret']).cumprod()
df['strat_cum']  = (1 + df['strat_ret_net']).cumprod()

# Quick peek at the end values
end_bh    = df['bh_cum'].iloc[-1]
end_strat = df['strat_cum'].iloc[-1]
end_bh, end_strat

# Visualize the trading performance
plt.figure(figsize=(12,6))
plt.plot(df['bh_cum'], label='Buy & Hold', color='black', linewidth=2)
plt.plot(df['strat_cum'], label='SMA Crossover Strategy', color='orange', linewidth=2)
plt.title('Cumulative Performance: AAPL (Buy & Hold vs SMA Strategy)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (Starting at $1)')
plt.legend()
plt.show()