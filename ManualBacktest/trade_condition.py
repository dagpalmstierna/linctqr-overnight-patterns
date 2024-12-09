from abc import ABC, abstractmethod
# Classes defining trade conditions. A list of conditions is entered as a parameter when initializing the main program

class TradeCondition(ABC):
    @abstractmethod
    def evaluate(self, current_row, prev_row, next_row, **kwargs):
        pass

class VolatilityCondition(TradeCondition):

    def __init__(self, upper_bound, lower_bound):
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, current_row, prev_row, next_row, **kwargs):
        return float(current_row["Volatility"].replace(",", ".")) < self.upper_bound and float(current_row["Volatility"].replace(",", ".")) > self.lower_bound

class CloseToCloseDownCondition(TradeCondition):
    def evaluate(self, current_row, prev_row, next_row, **kwargs):
        return prev_row["Close"] > current_row["Close"]

class NDayDownCondition(TradeCondition):
    def __init__(self, n):
        self.n = n

    def evaluate(self, current_row, prev_row, next_row, **kwargs):
        rows = kwargs.get("rows", [])
        if len(rows) < self.n:
            return False  # Not enough data to evaluate

        return all(row["Open"] > row["Close"] for row in rows[-self.n:])
    
class DailyLowCondition(TradeCondition):
    def evaluate(self, current_row, prev_row, next_row, **kwargs):
        low_price = current_row["Low"]
        close_price = current_row["Close"]
        
        threshold = low_price * 0.01

        return close_price <= low_price + threshold

class BigDropCondition(TradeCondition): 
    def evaluate(self, current_row, prev_row, next_row, **kwargs):
        open_price = current_row["Open"]
        close_price = current_row["Close"]
        
        threshold = 0.995
        return ( close_price / open_price <= threshold)
    
class WeekdayCondition(TradeCondition):
    WEEKDAY_MAP = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }

    def __init__(self, weekday):
        self.weekday = self.WEEKDAY_MAP.get(weekday.lower())
        if self.weekday is None:
            raise ValueError(f"Invalid weekday: {weekday}")

    def evaluate(self, current_row, prev_row, next_row, **kwargs):
        
        return current_row["Date"].weekday() == self.weekday
