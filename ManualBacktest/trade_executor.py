# Class to execute trades. Takes a performance tracker and strategy as parameters. Iterates through a dataframe and records the trades through the tracker

class TradeExecutor:
    def __init__(self, strategy, tracker):
        self.strategy = strategy
        self.tracker = tracker

    def execute(self, data, **kwargs):

        for ticker, df in data.items():
            recent_rows=[]
            for i in range(1, len(df) - 1):
                current_row = df.iloc[i]
                prev_row = df.iloc[i - 1]
                next_row = df.iloc[i + 1]

                recent_rows.append(current_row)
                if len(recent_rows) > self.strategy.max_lookback:
                    recent_rows.pop(0)

                if self.strategy.should_trade(current_row, prev_row, next_row, rows=recent_rows, **kwargs):
                    profit = (next_row["Open"])  -  (current_row["Close"])
                    self.tracker.record_trade(ticker, current_row["Close"], 100 * profit)

        return self.tracker.get_summary()
