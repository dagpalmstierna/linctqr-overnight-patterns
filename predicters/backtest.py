import backtrader as bt


class RandomForestStrategy(bt.Strategy):
    """
    Trading strategy that acts on Random Forest reversal predictions.
    Buys at close when a reversal is predicted, sells at next open.
    Position size is scaled inversely by volatility (VIX).
    """

    params = (
        ('predictions', None),
        ('volatility', None),
    )

    def __init__(self):
        self.predictions = self.p.predictions
        self.volatility = self.p.volatility
        self.current_day = None

    def next(self):
        current_date = self.data.datetime.date(0)

        # Sell any open position at the start of a new day (capturing the overnight return)
        if self.current_day != current_date:
            self.current_day = current_date
            if self.position:
                self.log(f"Selling at open")
                self.sell()

        # Buy at close if the model predicts a reversal and volatility is moderate
        if not self.position:
            if current_date in self.predictions and self.predictions[current_date] == 1:
                vol = self.volatility.get(current_date, None)
                if vol is not None and 0 < vol <= 30:
                    cash = self.broker.getcash()
                    price = (1 / vol) * cash
                    self.log(f"Buying at close (vol={vol:.1f})")
                    self.buy(price=price)

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime.date(0)
        print(f"{dt.isoformat()} - {txt}")
