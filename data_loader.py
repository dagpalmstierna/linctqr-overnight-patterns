import yfinance as yf
import pandas as pd
# Class to fetch data

class DataLoader:
    def __init__(self, tickers, start=None, end=None, external_data=None):
        self.tickers = tickers
        self.start = start
        self.end = end

        self.external_data = {}
        if external_data:
            for name, df in external_data.items():
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
                self.external_data[name] = df

    def process_file(self, file):
        df = pd.read_csv(file)
        df["Date"] = pd.to_datetime(df["Date"])


    def load_data(self):
        data = {}

        for ticker in self.tickers:
            ticker_data = yf.Ticker(ticker).history(start=self.start, end=self.end)
            
            ticker_data.reset_index(inplace=True)
            ticker_data["Date"] = pd.to_datetime(ticker_data["Date"])
            ticker_data["Date"] = ticker_data["Date"].dt.tz_localize(None)

            # Merge each external dataset with the ticker data
            
            for name, ext_df in self.external_data.items():
                ticker_data[name] = ticker_data["Date"].map(ext_df["Close"])

            # Handle missing values if necessary
            for name in self.external_data.keys():
                if name in ticker_data:
                    ticker_data[name] = ticker_data[name].fillna(method="ffill")  # Forward-fill missing values
            # Save the processed data
            data[ticker] = ticker_data

        
        return data

