from abc import ABC, abstractmethod
import pandas as pd 

class TradeCondition(ABC):
    @abstractmethod
    def should_trade(self, date: pd.Timestamp, open_price: float, close_price: float, openNextDay: float, previous_day_close: float, previous_day_open: float) -> bool:
        pass

class MondayCondition(TradeCondition):
    def should_trade(self, date: pd.Timestamp, open_price: float, close_price: float, openNextDay: float, previous_day_close: float, previous_day_open: float) -> bool:
        return date.weekday() == 0

class ReversalCondition(TradeCondition):
    def should_trade(self, date: pd.Timestamp, open_price: float, close_price: float, openNextDay: float, previous_day_close: float, previous_day_open: float) -> bool:
        return (open_price > close_price)
    
class TwoDayDown(TradeCondition):
    def should_trade(self, date: pd.Timestamp, open_price: float, close_price: float, openNextDay: float, previous_day_close: float, previous_day_open: float) -> bool:
        return (previous_day_close < previous_day_open)








        