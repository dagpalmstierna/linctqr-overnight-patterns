import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
import securities

from backtesting.test import SMA, GOOG
from backtesting.lib import crossover, plot_heatmaps
import talib

import yfinance as yf

import warnings
warnings.filterwarnings("ignore")


print(GOOG)

start_date = "2010-09-03"
end_date = "2024-11-12" 

data = yf.download("IHYG.L", 
                  start=start_date, 
                  end=end_date, 
                  auto_adjust=True, 
                  multi_level_index=False)

print(data)
#calculates based on close value and buys based in the valye next days open

def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()

class ReversalPatternStrategy(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    close_price, 


  
    def init(self):
        # Track whether we should be buying after a down day
        self.ready_to_sell_next_open = False
    
    def previousReversal(self):
      if self.data.Close[-3] > self.data.Open[-3] and self.data.Open[-2] > self.data.Close[-3]:
          return True

    def next(self):
        if self.data.Open[-2] > self.data.Close[-2]:
            # Buy at the close price of the current day
            self.trade_on_close = True; 
            self.buy(size=self.position.size, )

        # If there is an open position, sell at the open price of the next day
        if self.position:
            self.sell(size=self.position.size, price=self.data.Open[-1])
         
            
        # 3. Sell at the next day's open, unconditionally if in position
        
   
        
        

bt = Backtest(data, ReversalPatternStrategy, cash=10_000, commission=.002)
stats = bt.run()


