import pandas as pd
from randomforest import RandomForestModel
from dataloader import DataLoader
from backtest import RandomForestStrategy
import yfinance as yf
import backtrader as bt

dl = DataLoader()
rf1 = RandomForestModel()

rf2 = RandomForestModel()
    

def run_predicter():
   
    rf1.run_model()                  
    rf1.get_overall_results()        

   
    top_n = rf1.get_top_features()

    print("\n===== Second pass: using ONLY top n features =====")
    rf2.run_model(top_features=top_n)   
    rf2.get_overall_results()     
    
def run_backtest(model):
    df = model.df
    
    volume_series = yf.Ticker("IHYG.L").history(start=dl.start, end=dl.end)["Volume"]
    df["Volume"] = volume_series.reindex(df.index).fillna(0)

    predictions = { key.date(): value for key, value in zip(df.index, model.get_predictions())} 

    volatility = { key.date(): value for key, value in zip(df.index, df["Volatility"]) }

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    data_feed = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(1000000)

    cerebro.broker.set_coo(True)
    cerebro.broker.set_coc(True)

    cerebro.addstrategy(RandomForestStrategy, predictions=predictions, volatility=volatility)

    cerebro.run()
    cerebro.plot()







run_predicter()
run_backtest(rf1)




