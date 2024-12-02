from data_loader import DataLoader
import pandas as pd
import yfinance as yf
import numpy as np
from skopt import BayesSearchCV

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


class PredictionModel:
    def __init__(self, dataloader, ticker):
        self.df = dataloader.load_data()[ticker]
        
    # Method to clean and prepare data for the training and tests 
    def clean_data(self):
        df = self.df
        start="2010-09-03"
        end="2024-11-22"
        stoxx_df = yf.Ticker("^STOXX").history(start=start, end=end)
        stoxx_df.reset_index(inplace=True) 
        stoxx_df["Market"] = (1  - stoxx_df["Close"] / stoxx_df["Open"]) * 100
        df["Market"] = stoxx_df["Market"]

        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop rows with NaN
        df.dropna(inplace=True)
        df.drop("Date", axis=1, inplace=True)
        df.drop("Dividends", axis=1, inplace=True)
        df.drop("Capital Gains", axis=1, inplace=True)
        df.drop("Stock Splits", axis=1, inplace=True)
        df.drop("Volume", axis=1, inplace=True)
        
        
        df["Movement"] = (1 - df["Close"] / df["Open"]) 
        
        df["Day(s) down"] = df.apply(
            lambda x: 2 if x["Close"] < x["Open"] and df["Close"].shift(1).iloc[x.name] < df["Open"].shift(1).iloc[x.name]
        else 1 if x["Close"] < x["Open"]
        else 0,
        axis=1,
    )   
        df["Reversal"] = df.apply(lambda x: 1 if x["Close"] < df["Open"].shift(-1).iloc[x.name] and x["Close"] < x["Open"] else 0, axis=1)

        #NOTE not actual beta. Want to find actual beta through gathering hourly data and calculating 
        df["Beta"] = (1 - (df["High"] - df["Low"]) / df["Low"]) 
        
        df["Threshold"] = ((df["Close"] - df["Low"]) / df["Low"]) 

        df.drop("Open", axis=1, inplace=True)
        df.drop("Close", axis=1, inplace=True)
        df.drop("High", axis=1, inplace=True)
        df.drop("Low", axis=1, inplace=True)
        self.df = df
       

    # Returns a dataframe of the features with the highest significance. Might be necessary to filter if we were to add several more variables
    def importance(self):

        df = self.df
        y = df["Reversal"]
        X = df.drop("Reversal", axis=1)
        
        random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        random_forest.fit(X, y)


        feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': random_forest.feature_importances_})

        feature_df = feature_df.sort_values(by="Importance", ascending=False)
        return feature_df

    def test_model(self):
        
        df = self.df
        y = df["Reversal"].to_numpy()
        
        X = pd.get_dummies(df.drop("Reversal", axis=1))

        column_names = X.columns.tolist()
        X = X.to_numpy()
        
        
        #Test size 20 % 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_space = {
            'n_estimators': (50, 300),               # Number of trees
            'max_depth': (5, 20),                    # Depth of trees
            'min_samples_split': (2, 10),           # Minimum samples to split a node
            'min_samples_leaf': (1, 5),             # Minimum samples per leaf
            'max_features': ['sqrt', 'log2', None]  # Feature selection for splits
        }

        # Hyperparameter optimization through BayesSearch 
        # BayesSearchCV uses smarter guesses to find the best hyperparameters by learning from each test.
        # It tries a mix of exploring new options and improving known good ones. Runs 30 tests with 5-fold CV.

        opt = BayesSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        search_spaces=param_space,
        n_iter=30,  # Number of parameter combinations to try
        cv=5,       # 5-fold cross-validation
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        verbose=1   # Print progress
    )

        # Fit the model
        print("Still works")
        print("Starting hyperparameter tuning...")
        opt.fit(X_train, y_train)
        
    
        print("Best parameters:", opt.best_params_)
        print("Best cross-validation accuracy:", opt.best_score_)

        # Evaluate on test set
        y_pred = opt.best_estimator_.predict(X_test)
        print(classification_report(y_test, y_pred))

        y_pred = opt.best_estimator_.predict(X_test)
        print("Accuracy on test set:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        feature_importance = pd.DataFrame({
            'Feature': column_names,
            'Importance': opt.best_estimator_.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print("Feature importance")
        print(feature_importance)


