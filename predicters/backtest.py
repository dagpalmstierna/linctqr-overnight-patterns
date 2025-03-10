import backtrader as bt

class RandomForestStrategy(bt.Strategy):
    
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
        if self.current_day != current_date:
            self.current_day = current_date
        
            if self.position:
                self.log(f"New day ({current_date}): Selling position at OPEN")
                self.sell() 
               
        if not self.position:
            if current_date in self.predictions and self.predictions[current_date] == 1:
                vol = self.volatility.get(current_date, None)
                print(vol)
                if vol is not None and vol <= 30 and vol > 0:
                    cash = self.broker.getcash()
                    price = (1/vol) * cash
                    self.log(f"Date is ({current_date}) signal is 1 and volatility is {vol}<=30: Buying at CLOSE: {cash}")
                    self.buy(price=price)
                               
    def log(self, txt, dt=None):
        dt = dt or self.data.datetime.date(0)
        print(f"{dt.isoformat()} - {txt}")


