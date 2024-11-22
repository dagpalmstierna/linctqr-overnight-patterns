
import pandas as pd

class PerformanceTracker:
    def __init__(self):
        self.volume = 0
        self.nbr_of_trades = 0
        self.profit = 0
        self.rate = 1
        self.nbr_of_reversals = 0
        self.red_days = 0

    def record_trade(self, volume, profit):
        self.volume += volume
        self.profit += profit
        self.rate = (self.rate + (profit/volume)) / self.rate
        self.nbr_of_trades += 1
        

    def record_reversal(self):
        self.nbr_of_reversals += 1

    def record_red_day(self):
        self.red_days += 1

    def get_summary(self):
        data = {
            "Number of reversals": [self.nbr_of_reversals],
            "Traded Volume": [self.volume],
            "Profit": [self.profit],
            "Result %": [(self.rate * 100)],
            #"Number of Trades": [self.nbr_of_trades],
            #"Number of Reversals": [self.nbr_of_reversals],
            #"Reversal Ratio": [self.nbr_of_reversals / self.red_days] if self.red_days > 0 else [0],
        }
        return pd.DataFrame(data)

  