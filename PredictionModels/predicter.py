import pandas as pd
import yfinance as yf
import numpy as np
from skopt import BayesSearchCV # RandomizedSearchCV, GridSearchCV

from abc import ABC, abstractmethod
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

     
#TODO XGBoost
#lightgbm
#logistic regression
#SVM

class PredictionModel(ABC):
    
    def __init__(self):
        
        self.df = pd.DataFrame()
        self.start ="2010-09-03"
        self.end="2024-11-22"
        self.fetch_data()
    
    def fetch_data(self):

        self.dict = {"ETF data": self.get_etf_data("IHYG.L"),
                    "Stock market": self.get_market_data(),
                    "Volatility": self.get_volatility_data(),
                    "Interest rate": self.get_interest_rates(),
                    "Inflation": self.get_inflation_levels()}
    
    def get_etf_data(self, ticker):
        return yf.Ticker(ticker).history(start=self.start, end=self.end)
    
    def get_market_data(self):
        return yf.Ticker("^STOXX").history(start=self.start, end=self.end)
    
    def get_volatility_data(self):
        return pd.read_csv("Indices/VSTOXX.csv")
    
    def get_interest_rates(self):
        return pd.read_csv("Indices/ECB Interest rates.csv").drop(columns="TIME PERIOD")
    
    def get_inflation_levels(self):
        inflation_df = pd.read_csv("Indices/ECB HICP.csv").drop(columns="Time period")

        inflation_df["Date"] = pd.to_datetime(inflation_df["Date"])
        inflation_df["Month"] = inflation_df["Date"].dt.month
        inflation_df["Year"] = inflation_df["Date"].dt.year
        
        
        daily_dates = pd.date_range(start=self.start, end=self.end)
        daily_df = pd.DataFrame({"Date": daily_dates})

        
        daily_df["Month"] = daily_df["Date"].dt.month
        daily_df["Year"] = daily_df["Date"].dt.year


        daily_inflation_df = daily_df.merge(
            inflation_df[["Year", "Month", "Inflation"]],
            on=["Year", "Month"],
            how="left"
        )
    
        return daily_inflation_df

    def clean_data(self):

        for key in self.dict.keys():
            df = self.dict[key]

            # Om datum finns som index, flytta det till en kolumn
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()

            # Kontrollera om 'Date'-kolumnen existerar innan bearbetning
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
                df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
                
            self.dict[key] = df
    
        self.add_column("Reversal")
        self.add_column("Market performance")
        self.add_column("ETF performance")
        self.add_column("Days down")
        self.add_column("Threshold")
        self.add_column("Deposit facility")
        self.add_column("Marginal lending facility")
        self.add_column("Main refinancing operations")
        self.add_column("Inflation level")
        self.add_column("Season")
        self.add_column("Weekday")
        
        
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
    
    def add_column(self, name):
        etf_df = self.dict.get("ETF data")
        interest_rates_df = self.dict.get("Interest rate")
        
        match name:
            case "Reversal":
                self.df[name] = etf_df.apply(lambda x: 1 if self.isReversal(x["Close"], etf_df["Open"].shift(-1).iloc[x.name], x["Open"]) else 0, axis=1)
            case "Market performance":
                stoxx_df = self.dict.get("Stock market")
                self.df[name] = (stoxx_df["Close"] / stoxx_df["Open"]) 
            case "ETF performance":
                self.df[name] = (etf_df["Close"]/ etf_df["Open"]) 
            case "Days down":
                days_down = [0]
                movement_list = self.df["ETF performance"]
                for i in range(1, len(movement_list)):
                    if movement_list[i - 1] < 1:  
                        days_down.append(days_down[-1] + 1)  
                    else:
                        days_down.append(0)
                self.df[name] = days_down
            case "Threshold":
                self.df[name] = etf_df["Close"] / etf_df["Low"] 
            case "Deposit facility":
                self.df[name] = interest_rates_df[name]
            case "Marginal lending facility":
                self.df[name] = interest_rates_df[name]
            case "Main refinancing operations":
                self.df[name] = interest_rates_df[name]
            case "Inflation level":
                self.df[name] = self.get_inflation_levels()["Inflation"]
            case "Season":
                self.df[name] = etf_df["Date"].dt.month.map(self.get_season)
            case "Weekday":
                self.df[name] = etf_df["Date"].dt.weekday
            

    
    def get_season(self, month):
        match month:
            case 12 | 1 | 2:
                return 1  # Vinter
            case 3 | 4 | 5:
                return 2  # Vår
            case 6 | 7 | 8:
                return 3  # Sommar
            case 9 | 10 | 11:
                return 4  # Höst
            
    def isReversal(self, closeprice, nextopenprice, openprice):
        return (closeprice < nextopenprice and closeprice < openprice) or (closeprice > nextopenprice and closeprice > openprice)
    
    @abstractmethod
    def importance(n):
        pass

    
    def test_model(self):
        self.perform_tests()
    
    @abstractmethod
    def perform_tests():
        pass