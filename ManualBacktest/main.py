
import pandas as pd
from ManualBacktest.data_loader import DataLoader
from ManualBacktest.trade_condition import VolatilityCondition, NDayDownCondition, BigDropCondition, DailyLowCondition, CloseToCloseDownCondition, WeekdayCondition
from ManualBacktest.trade_strategy import TradeStrategy
from ManualBacktest.trade_executor import TradeExecutor
from ManualBacktest.performance_tracker import PerformanceTracker
import yfinance as yf

def run_tradingStrategy(tickers, conditions):
    tracker = PerformanceTracker()
    executor = TradeExecutor(TradeStrategy(conditions=conditions), tracker=tracker)

    # Fetch the needed data
    volatility_df = pd.read_csv("VSTOXX.csv")
    start="2010-09-03"
    end="2024-11-22"
    stoxx_df = yf.Ticker("^STOXX").history(start=start, end=end)
    stoxx_df.reset_index(inplace=True) 
    external_data = {"Volatility": volatility_df,
                     "Market Movement": stoxx_df}

    dataloader = DataLoader(tickers, start=start, end=end, external_data=external_data)
   
    # Simulate and print results
    print(executor.execute(dataloader.load_data()))
    

# Example strategy with 2-day-down condition, Volatility between 15 - 20, a drop of at least 0.5 % and a close level within a 1 % treshold of daily low
run_tradingStrategy(["IHYG.L", "SYBJ.DE"], [NDayDownCondition(2), BigDropCondition(), DailyLowCondition(), VolatilityCondition(20, 15)])

# Running prediction model on iShares High Yield ETF historical data
#run_predicter("IHYG.L")





















