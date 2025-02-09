
import pandas as pd
import yfinance as yf
import talib as ta

class DataLoader():

    def __init__(self, start="2014-01-31", end="2024-11-22"):
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.load_data()
    
    def load_data(self):
        df = self.get_etf_data()
        
        #Intradaily data
        df["Credit spread"] = self.get_benchmark("ICE BoFa High Yield Index OAS")["Close"]
        self.add_extra_features(df, "Credit spread")
        self.add_ta_features(df, "Credit spread")


        df["CDS"] = self.get_benchmark("ITRAXX")["Close"]
        self.add_extra_features(df, "CDS")
        self.add_ta_features(df, "CDS")

        df["Yield curve spread"] = self.get_benchmark("ECB Yield curve spread")["Close"]
        self.add_extra_features(df, "Yield curve spread")
        self.add_ta_features(df, "Yield curve spread")

        #df["IR swap"] = self.get_benchmark("EUSA5")["Close"]
        #self.add_extra_features(df, "IR swap")
        #self.add_ta_features(df, "IR swap")

        #Interdaily data, has more parameters
        vstoxx_df = self.get_benchmark("VSTOXX")
        df["Volatility"] = vstoxx_df["Close"]
        self.add_extra_features(df, "Volatility")
        df["Volatility return"] = (1 - vstoxx_df["Close"] / vstoxx_df["Open"]) * 100
        self.add_ta_features(df, "Volatility")

        df["Stock returns"] = self.get_equity_data()
        self.add_extra_features(df, "Stock returns")
        self.add_ta_features(df, "Stock returns")
        
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(df.mean())
        df = df = df[df["Close"] < df["Open"]]

        #Endast dagar då ETFen går ner, dvs. då reversals faktiskt kan ske
        df = df[df["Close"] < df["Open"]]
        df.drop(columns=["Dividends", "Stock Splits", "Volume", "Capital Gains"], inplace=True)
        self.df = df
      

    def add_extra_features(self, df, col="Close"):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(method='ffill')
        df[f"{col} deviation"] = df[col].rolling(window=5).std().fillna(df[col].mean())
        df[f"{col} return"] = df[col].pct_change(fill_method=None)
        df[f"{col} lag"] = df[col].shift(1).fillna(df[col].mean())
       
    def add_ta_features(self, df, col):
        
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(method='ffill').fillna(method='bfill')

        # RSI: Beräknar Relative Strength Index
        df[f'{col} RSI'] = ta.RSI(df[col], timeperiod=14).fillna(method='ffill').fillna(method='bfill')
      
        # MACD: Returnerar MACD, signallinje och MACD-histogram
        macd, macdsignal, macdhist = ta.MACD(df[col],
                                            fastperiod=12,
                                            slowperiod=26,
                                            signalperiod=9)
        df[f'{col} MACD'] = macd
        df[f'{col} MACD_signal'] = macdsignal
        df[f'{col} MACD_diff'] = macdhist

        # Bollinger Bands: Returnerar övre, mitt och nedre band
        upperband, middleband, lowerband = ta.BBANDS(df[col],
                                                    timeperiod=20,
                                                    nbdevup=2,
                                                    nbdevdn=2,
                                                    matype=0)
        df[f'{col} bb_upper'] = upperband.fillna(method='ffill').fillna(method='bfill')
        df[f'{col} bb_middle'] = middleband.fillna(method='ffill').fillna(method='bfill')
        df[f'{col} bb_lower'] = lowerband.fillna(method='ffill').fillna(method='bfill')
       
    def get_etf_data(self, src="IHYG.L"):
        df = yf.Ticker(src).history(start=self.start, end=self.end)
        next_open = df["Open"].shift(-1)
        reversal = ((df["Close"] < next_open) & (df["Close"] < df["Open"])) 

        df["ETF return"] = (df["Close"] / df["Open"] - 1) 
        df["CLV"] = ((df["Close"] - df["Low"]) - (df["High"] - df["Low"])) /  (df["High"] - df["Low"])

        #reversal_movement = np.where(df["Close"] < df["Open"], (next_open / df["Close"] - 1) * 100, 0)
        #df["ReversalMovement"] = reversal_movement
        df["Reversal"] = reversal

        df["Date"] = df.index.tz_localize(None)
        df.reset_index(drop=True, inplace=True)
        self.sort_dates(df)
        df = df.loc[self.start:self.end]
        return df
    
    def get_equity_data(self):
        df = yf.Ticker("^STOXX").history(start = self.start, end = self.end)
        df["Date"] = df.index.tz_localize(None)
        df.reset_index(drop=True, inplace=True)
        self.sort_dates(df)
        df = df.loc[self.start:self.end]
        return (1 - df["Close"] / df["Open"]) * 100

    def get_benchmark(self, src): 
        df = pd.read_csv(f"data/{src}.csv", converters={"Close": lambda x: str(x.replace(",", "."))}, decimal=",")

        self.sort_dates(df)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.loc[self.start:self.end] 
        return df
    
    def sort_dates(self, df):
        df["Date"] = pd.to_datetime(df["Date"],infer_datetime_format=True, errors="coerce")
        df.set_index("Date", inplace=True)
        df.sort_index(ascending=True, inplace=True)
    
    def apply_pca(self, n_components=0.95):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        features = self.df.drop(columns=["Reversal"])
        features = features.apply(pd.to_numeric, errors='coerce')
        features = features.fillna(features.mean())

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(scaled_features)

        pca_df = pd.DataFrame(data=pca_features, columns=[f"PC{i+1}" for i in range(pca_features.shape[1])])
        pca_df["Reversal"] = self.df["Reversal"]
        self.df = pca_df
    
    def get_data(self):
        return self.df


