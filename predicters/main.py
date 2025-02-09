import pandas as pd
from predicter import RandomForestModel
from dataloader import DataLoader
from backtest import RandomForestStrategy
import yfinance as yf
import backtrader as bt

dl = DataLoader()
rf = RandomForestModel()

def run_predicter():
    rf.run_model()
    rf.get_overall_results()

def run_backtest():
    df = rf.df
    print(df)
    volume_series = yf.Ticker("IHYG.L").history(start=dl.start, end=dl.end)["Volume"]
    df["Volume"] = volume_series.reindex(df.index).fillna(0)

    # Hämta nyckelvärde tabell för prediktioner med datum som nycklar
    predictions = { key.date(): value for key, value in zip(df.index, rf.get_predictions())} 

    # Samma här fast med volatilitet
    volatility = { key.date(): value for key, value in zip(df.index, df["Volatility"]) }

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    data_feed = bt.feeds.PandasData(dataname=df)

    # Sätt upp Cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(100000)

    # Aktivera cheat on open/close för att kunna handla både öppning och stängning
    cerebro.broker.set_coo(True)
    cerebro.broker.set_coc(True)

    cerebro.addstrategy(RandomForestStrategy, predictions=predictions, volatility=volatility)

    results = cerebro.run()
    cerebro.plot()

run_predicter()
run_backtest()





