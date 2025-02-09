import backtrader as bt
import pandas as pd
import yfinance as yf
from predicter import RandomForestModel
from dataloader import DataLoader
import datetime

class RandomForestStrategy(bt.Strategy):
    params = (
        ('predictions', None),  # Dictionary: {datetime.date: 0 eller 1}
        ('volatility', None),   # Dictionary: {datetime.date: numeriskt värde}
    )

    def __init__(self):
        self.predictions = self.p.predictions
        self.volatility = self.p.volatility

        # Håll reda på vilket datum vi är på (för att upptäcka dagsskifte)
        self.current_day = None  
        # Flagga för att se om vi redan har sålt under den nya dagen
        self.sell_executed = False  

    def next(self):
        current_date = self.data.datetime.date(0)
        
        # Upptäck dagsskifte: om vi har gått in i en ny dag
        if self.current_day != current_date:
            self.current_day = current_date
            self.sell_executed = False  # Återställ flaggan för den nya dagen
            
            # Om vi har en position från tidigare, så sälj den vid öppningen (simulerat)
            if self.position:
                self.log(f"Ny dag ({current_date}): Säljer position (simulerar sälj vid OPEN)")
                self.sell()  # Denna order kommer exekveras med nästa möjliga pris (om cheat_on_open är aktiverat blir det öppningspriset)
                self.sell_executed = True

        # Vid barens stängning (eller när vi är inne i dagens next()-cykel) kontrollerar vi om vi ska köpa
        if not self.position:
            # Om dagens datum finns med i predictions och signalen är 1
            if current_date in self.predictions and self.predictions[current_date] == 1:
                vol = self.volatility.get(current_date, None)
                if vol is not None and vol <= 30:
                    self.log(f"Dagens ({current_date}) signal är 1 och vol {vol}<=30: Köper vid CLOSE")
                    self.buy()
                    
    def log(self, txt, dt=None):
        """ Enkel loggningsfunktion """
        dt = dt or self.data.datetime.date(0)
        print(f"{dt.isoformat()} - {txt}")



# dl = DataLoader()
# rf = RandomForestModel(dl)
# rf.run_model()
# df = rf.df
# volume_series = yf.Ticker("IHYG.L").history(start=dl.start, end=dl.end)["Volume"]
# df["Volume"] = volume_series.reindex(df.index).fillna(0)
# print(df["Volume"])
# print(volume_series)


# predictions = { key.date(): value for key, value in zip(df.index, rf.get_predictions())} 
# # Konvertera volatiliteten: här skapar vi en dictionary med nycklar i datetime.date-format
# volatility = { key.date(): value for key, value in zip(df.index, df["Volatility"]) }

# # Ladda historisk data med yfinance (se till att intervallet matchar dina dictionaries)

# if isinstance(df.columns, pd.MultiIndex):
#     df.columns = df.columns.get_level_values(0)

# data_feed = bt.feeds.PandasData(dataname=df)

# # Sätt upp Cerebro
# cerebro = bt.Cerebro()
# cerebro.adddata(data_feed)
# cerebro.broker.setcash(100000)

# # Aktivera cheat on open/close (om du vill att order ska exekveras precis vid öppning/stängning)
# cerebro.broker.set_coo(True)
# cerebro.broker.set_coc(True)

# # Lägg till strategin med dina dictionaries
# cerebro.addstrategy(RandomForestStrategy, predictions=predictions, volatility=volatility)

# results = cerebro.run()
# cerebro.plot()
