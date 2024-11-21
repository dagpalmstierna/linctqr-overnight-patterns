import yfinance as yf
import pandas as pd
from tracker import PerformanceTracker
from trade_condition import ReversalCondition, MondayCondition

class Security:
    def __init__(self, ticker: str, trade_conditions=None):
        self.ticker = ticker
        self.dataframe = yf.Ticker(self.ticker).history(period="max")
        self.performance_tracker = PerformanceTracker()
        self.trade_conditions = trade_conditions or []  # List of conditions to check

    def find_reversals(self):
        df = self.dataframe
        reversals = pd.DataFrame()
        df.index = pd.to_datetime(df.index)

        for i in range(len(df) - 1):
            open_price = df.iloc[i]["Open"]
            close_price = df.iloc[i]["Close"]
            open_next_day = df.iloc[i + 1]["Open"]
            date = df.index[i]

            if open_price > close_price:
                self.performance_tracker.record_red_day()
                if close_price < open_next_day:
                    self.performance_tracker.record_reversal()
                    reversals = pd.concat([reversals, df.iloc[[i]], df.iloc[[i + 1]]])

            if self.should_trade(date, open_price, close_price, open_next_day):
                self.performance_tracker.record_trade(close_price, (open_next_day - close_price))
        
        reversals.reset_index(drop=True, inplace=True)
        return reversals

    def should_trade(self, date, open_price, close_price, open_next_day):
        
        return all(condition.should_trade(date, open_price, close_price, open_next_day) for condition in self.trade_conditions)

    def get_performance_summary(self):
        return self.performance_tracker.get_summary()