# Sets the logic for the trade strategy. Initialized with a set number of conditions

class TradeStrategy:
    def __init__(self, conditions):
        self.conditions = conditions
        self.max_lookback = max([getattr(cond, 'n', 0) for cond in conditions])  

    def should_trade(self, current_row, prev_row, next_row, **kwargs):
        return all(condition.evaluate(current_row, prev_row, next_row, **kwargs) for condition in self.conditions)
