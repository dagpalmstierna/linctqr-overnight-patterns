
import pandas as pd
from data_loader import DataLoader
from trade_condition import VolatilityCondition, NDayDownCondition, BigDropCondition, DailyLowCondition, CloseToCloseDownCondition, WeekdayCondition
from trade_strategy import TradeStrategy
from trade_executor import TradeExecutor
from performance_tracker import PerformanceTracker
import yfinance as yf
from predicter import PredictionModel




#TODO: Implement logic for cds index and check for patterns
#TODO: Find further variables for the model 

start="2010-09-03"
end="2024-11-22"

def run_predicter(ticker):
    pm = PredictionModel(dataloader=DataLoader([ticker], start=start, end=end), ticker=ticker)
    pm.clean_data()
    pm.test_model()

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





















