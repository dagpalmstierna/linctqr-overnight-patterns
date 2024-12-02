import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy


from backtesting.test import SMA, GOOG
from backtesting.lib import crossover, plot_heatmaps
import talib

import yfinance as yf

import warnings
warnings.filterwarnings("ignore")

# TO-DO: 
# Customize dataframe with values that allows buying on close and selling on open 
# This is needed to backtest the strategy with inbuilt library 

class ReversalPatternStrategy(Strategy):
    
    def init(self):
        return 
    def next(self):
        return

    
