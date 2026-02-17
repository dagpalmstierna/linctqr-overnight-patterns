import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=FutureWarning, message=".*Series.fillna with 'method' is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*infer_datetime_format.*")


class DataLoader:

    def __init__(self, start="2010-09-03", end="2023-03-17"):
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.load_data()
        self.apply_pca()

    def load_data(self):
        df = self.get_etf_data(src="IHYG.L")

        df["Credit spread"] = self.get_benchmark("ICE BoFa High Yield Index OAS")["Close"]
        self.add_extra_features(df, "Credit spread")
        self.add_ta_features(df, "Credit spread")

        df["Itraxx crossover"] = self.get_itraxx_data("DBXM.DE")["Close"]
        df["Itraxx Europe"] = self.get_benchmark("ITRAXXETF")["Price"]
        self.add_extra_features(df, "Itraxx crossover")
        self.add_ta_features(df, "Itraxx crossover")
        self.add_extra_features(df, "Itraxx Europe")
        self.add_ta_features(df, "Itraxx Europe")

        df["Yield curve spread"] = self.get_benchmark("ECB Yield curve spread")["Close"]
        self.add_extra_features(df, "Yield curve spread")
        self.add_ta_features(df, "Yield curve spread")

        vstoxx_df = self.get_benchmark("VSTOXX")
        df["Volatility"] = vstoxx_df["Close"]
        self.add_extra_features(df, "Volatility")
        df["Volatility return"] = (1 - vstoxx_df["Close"] / vstoxx_df["Open"]) * 100
        self.add_ta_features(df, "Volatility")

        df["Stock returns"] = self.get_equity_data()
        self.add_extra_features(df, "Stock returns")
        self.add_ta_features(df, "Stock returns")

        df["VIX"] = self.get_vix_data()
        self.add_extra_features(df, "VIX")
        self.add_ta_features(df, "VIX")

        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(df.mean())

        # Keep only days where the ETF closed lower than it opened (potential reversal days)
        df = df[df["Close"] < df["Open"]]
        df.drop(columns=["Dividends", "Stock Splits", "Volume", "Capital Gains"], inplace=True)
        self.df = df

    def add_extra_features(self, df, col="Close"):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(method='ffill')
        df[f"{col} deviation"] = df[col].rolling(window=5).std().fillna(df[col].mean())
        df[f"{col} change"] = df[col].pct_change(fill_method=None)
        df[f"{col} lag"] = df[col].shift(1).fillna(df[col].mean())

    def add_ta_features(self, df, col):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(method='ffill').fillna(method='bfill')
        df[f'{col} RSI'] = ta.RSI(df[col], timeperiod=14).fillna(method='ffill').fillna(method='bfill')

        macd, macdsignal, macdhist = ta.MACD(df[col], fastperiod=12, slowperiod=26, signalperiod=9)
        df[f'{col} MACD'] = macd
        df[f'{col} MACD_signal'] = macdsignal
        df[f'{col} MACD_diff'] = macdhist

        upperband, middleband, lowerband = ta.BBANDS(df[col], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df[f'{col} bb_upper'] = upperband.fillna(method='ffill').fillna(method='bfill')
        df[f'{col} bb_middle'] = middleband.fillna(method='ffill').fillna(method='bfill')
        df[f'{col} bb_lower'] = lowerband.fillna(method='ffill').fillna(method='bfill')

    def fetch_yf_data(self, ticker):
        return yf.Ticker(ticker).history(start=self.start, end=self.end)

    def get_etf_data(self, src="SYBJ.DE"):
        df = self.fetch_yf_data(src)
        next_open = df["Open"].shift(-1)
        reversal = (df["Close"] < next_open) & (df["Close"] < df["Open"])

        df["ETF return"] = df["Close"] / df["Open"] - 1
        df["CLV"] = ((df["Close"] - df["Low"]) - (df["High"] - df["Low"])) / (df["High"] - df["Low"])
        df["Reversal"] = reversal

        self.sort_dates(df)
        return df

    def get_itraxx_data(self, ticker):
        df = self.fetch_yf_data(ticker)
        self.sort_dates(df)
        return df

    def get_vix_data(self):
        df = self.fetch_yf_data("^VIX")
        self.sort_dates(df)
        return df["Close"]

    def get_equity_data(self):
        df = self.fetch_yf_data("^STOXX")
        self.sort_dates(df)
        return (1 - df["Close"] / df["Open"]) * 100

    def get_benchmark(self, src):
        if src == "ITRAXXETF":
            df = pd.read_csv(
                f"data/{src}.csv",
                converters={"Price": lambda x: str(x.replace(",", "."))},
                parse_dates=["Date"],
                date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%Y'),
                decimal=","
            )
        else:
            df = pd.read_csv(
                f"data/{src}.csv",
                converters={"Close": lambda x: str(x.replace(",", "."))},
                decimal=","
            )

        self.sort_dates(df)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.loc[self.start:self.end]
        return df

    def sort_dates(self, df):
        if not isinstance(df.index, pd.DatetimeIndex):
            df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors="coerce")
            df.set_index("Date", inplace=True)

        df.index = df.index.tz_localize(None)
        df.sort_index(ascending=True, inplace=True)
        df = df.loc[self.start:self.end]

    def balance_training_data_with_time(self, train_df, window="6M"):
        """Apply SMOTE oversampling within rolling time windows to preserve temporal structure."""
        balanced_dfs = []

        for group_time, group in train_df.groupby(pd.Grouper(freq=window)):
            if len(group) == 0:
                continue

            X_group = group.drop(columns=["Reversal"])
            y_group = group["Reversal"]
            X_reset = X_group.reset_index(drop=True)
            y_reset = y_group.reset_index(drop=True)

            minority_count = y_reset.value_counts().min()

            if minority_count < 2:
                group_balanced = group.copy()
            else:
                n_neighbors = min(5, minority_count - 1)
                smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
                try:
                    X_res, y_res = smote.fit_resample(X_reset, y_reset)
                    new_index = pd.RangeIndex(start=0, stop=len(X_res))
                    group_balanced = pd.DataFrame(X_res, columns=X_group.columns, index=new_index)
                    group_balanced["Reversal"] = y_res.astype(int)
                except Exception as e:
                    print(f"SMOTE failed for window {group_time}: {e}")
                    group_balanced = group.copy()

            balanced_dfs.append(group_balanced)

        return pd.concat(balanced_dfs, ignore_index=True)

    def apply_pca(self, n_components=0.95):
        """Reduce dimensionality with PCA, retaining 95% of explained variance."""
        df = self.df.copy()
        features = df.drop(columns=["Reversal"])
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features = features.apply(pd.to_numeric, errors='coerce')
        features = features.fillna(features.mean())

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        pca = PCA()
        pca.fit(scaled_features)

        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(explained_variance >= n_components) + 1

        pca = PCA(n_components=num_components)
        pca_features = pca.fit_transform(scaled_features)

        pca_df = pd.DataFrame(
            data=pca_features,
            columns=[f"PC{i+1}" for i in range(pca_features.shape[1])]
        )
        pca_df.index = df.index
        pca_df["Reversal"] = self.df["Reversal"].astype(int)

        self.df = pca_df

    def get_data(self):
        return self.df
