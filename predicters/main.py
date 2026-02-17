import pandas as pd
import yfinance as yf
import backtrader as bt
from dataloader import DataLoader
from randomforest import RandomForestModel
from backtest import RandomForestStrategy

dl = DataLoader()
rf = RandomForestModel(dl)


def run_predictor():
    """Two-pass approach: run with all features, then re-run using only the top-n most important."""
    rf.run_model()
    rf.get_overall_results()

    top_n = rf.get_top_features()
    print("\n===== Second pass: using only top features =====")

    rf2 = RandomForestModel(dl)
    rf2.run_model(top_features=top_n)
    rf2.get_overall_results()


def run_backtest():
    """Backtest the model predictions using backtrader."""
    df = rf.df.copy()

    volume_series = yf.Ticker("IHYG.L").history(start=dl.start, end=dl.end)["Volume"]
    df["Volume"] = volume_series.reindex(df.index).fillna(0)

    predictions = {key.date(): value for key, value in rf.get_predictions().items()}
    volatility = {key.date(): value for key, value in zip(df.index, df.get("Volatility", pd.Series(dtype=float)))}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    data_feed = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(1_000_000)
    cerebro.broker.set_coo(True)
    cerebro.broker.set_coc(True)
    cerebro.addstrategy(RandomForestStrategy, predictions=predictions, volatility=volatility)

    cerebro.run()
    cerebro.plot()


if __name__ == "__main__":
    run_predictor()
    run_backtest()
