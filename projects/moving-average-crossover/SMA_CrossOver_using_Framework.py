import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

# Download data
ticker = "AAPL"
data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True)  
data = data[['Open','High','Low','Close','Volume']].dropna().sort_index()
data.head()

# Define SMA Strategy
def EMA(x, n):
        return pd.Series(x).ewm(span=n, adjust=False).mean()
    
class SmaCross(Strategy):
    fast = 20
    slow = 50

    # You can either use the exponential moving average (EMA) or the simple moving average (SMA)
    def init(self):
        price = self.data.Close
        #self.sma_fast = self.I(SMA, price, self.fast)
        #self.sma_slow = self.I(SMA, price, self.slow)
        self.ema_fast = self.I(EMA, price, self.fast)
        self.ema_slow = self.I(EMA, price, self.slow)

    def next(self):
        if crossover(self.ema_fast, self.ema_slow):
            self.position.close()   # close any short
            self.buy()
        elif crossover(self.ema_slow, self.ema_fast):
            self.position.close()
            #self.sell() 

data = data.droplevel(1, axis=1)

# Backtest the strategy and visualize it's performance
bt = Backtest(
    data,
    SmaCross,
    cash=10_000,
    commission=0.001,        # 0.10% per trade
    trade_on_close=True,     # act after the bar’s signal
    exclusive_orders=True,
    finalize_trades=True
)

stats = bt.run()
stats
bt.plot()