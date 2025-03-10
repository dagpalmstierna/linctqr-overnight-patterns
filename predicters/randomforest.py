from sklearn.svm import SVC
from dataloader import DataLoader
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, recall_score
import seaborn as sns
import numpy as np
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

class RandomForestModel: 
    def __init__(self, dl=DataLoader(), tune=False):
        self.dl = dl
        self.df = dl.get_data()
        self.results={}
        self.tune = tune
        self.aggregated_importances = None
        self.predictions = {}

    def run_model(self, top_features=None):
        df = self.df
        X = df.drop(columns=["Reversal"])
        
        if top_features != None: 
            X = X[top_features]
            
        #Extra cleaning
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(method='ffill', inplace=True)
        X = X.astype(np.float64)
       
        y = df["Reversal"].astype(int)

        # Time series cross validation 
        tscv = TimeSeriesSplit(n_splits=5)
    
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]
           
            
            print(f"Fold {fold + 1}")
            print("Training data time period:", train_df.index[0], "to", train_df.index[-1])
            print("Test data time period:", test_df.index[0], "to", test_df.index[-1])
            
            # Här balanserar vi ENBART träningsdatan med tidsbaserad SMOTE
            train_df_balanced = self.dl.balance_training_data_with_time(train_df)

            # Separera features och target efter balansering
            X_train = train_df_balanced.drop(columns=["Reversal"])

            y_train = train_df_balanced["Reversal"].astype(int)
            print(y_train.value_counts())
            X_test = test_df.drop(columns=["Reversal"])
            y_test = test_df["Reversal"].astype(int)
            print(f"Fold {fold + 1}")
            

            #Hypertuning
            if self.tune:
                param_grid = {
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
                base_model = RandomForestClassifier(random_state=42)
                grid = GridSearchCV(base_model, param_grid, cv=3, scoring='f1', n_jobs=-1)

                grid.fit(X_train, y_train)
                rf_model = grid.best_estimator_
                print("Bästa parametrar för fold", fold+1, ":", grid.best_params_)
            else:
                # Använd tidigare tunade parametrar för 5 folds
                rf_model = RandomForestClassifier(random_state=42, max_depth=10, min_samples_split=2, max_features=None, n_estimators=100)
                rf_model.fit(X_train, y_train)
            
            # Gör prediktioner på testdatan
            rf_probs = rf_model.predict_proba(X_test)[:, 1]
            rf_predictions = (rf_probs > 0.5).astype(int)
            
            # svm_model = SVC(probability=True, random_state=42)
            # svm_model.fit(X_train, y_train)
            # svm_probs = svm_model.predict_proba(X_test)[:, 1]
            # svm_predictions = (svm_probs > 0.5).astype(int)

            # Kombinera prediktionerna
            combined_predictions = rf_predictions #np.logical_and(rf_predictions, svm_predictions).astype(int)


            acc = accuracy_score(y_test, combined_predictions)
            f1 = f1_score(y_test, combined_predictions)
            recall = recall_score(y_test, combined_predictions)
            conf_matrix = confusion_matrix(y_test, combined_predictions)
            importances = rf_model.feature_importances_
            feature_names = X_train.columns
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values(by="importance", ascending=False)


            # Spara resultatet för denna fold
            #predictions = { key.date(): value for key, value in zip(etf_data.index, rf1.get_predictions())} 

            # self.predictions[] = {
            #     "predictions": predictions
            # }

            #sdates = test_df.index.tz_localize(None)
            pred_dates = pd.DatetimeIndex(test_df.index.tz_localize(None))           
            fold_pred_dict = {date: pred for date, pred in zip(pred_dates, combined_predictions)}

            self.results[fold] = {
                "predictions": combined_predictions,
                "predictions_with_dates": fold_pred_dict,
                "accuracy": acc,
                "f1": f1,
                "recall": recall,
                "confusion_matrix": conf_matrix,
                "model": rf_model,
                "feature_importances": importance_df
            }


            print(f"Fold {fold + 1} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}\n")

        #self.get_overall_results()
        
    def get_overall_results(self):
        
        overall_acc = sum([self.results[fold]["accuracy"] for fold in self.results]) / len(self.results)
        overall_f1 = sum([self.results[fold]["f1"] for fold in self.results]) / len(self.results)
        overall_recall = sum([self.results[fold]["recall"] for fold in self.results]) / len(self.results)
        total_conf_matrix = np.zeros_like(list(self.results.values())[0]["confusion_matrix"])

        for fold in self.results:
            total_conf_matrix += self.results[fold]["confusion_matrix"]

        all_importances = [self.results[fold]["feature_importances"] for fold in self.results]
        importances_concat = pd.concat(all_importances)
        self.aggregated_importances = (importances_concat.groupby('feature', as_index=False)['importance']
                                       .mean()
                                       .sort_values(by="importance", ascending=False))
        
        print(f"Overall accuracy: {overall_acc:.4f}")
        print(f"Overall F1-score: {overall_f1:.4f}")
        print(f"Overall Recall: {overall_recall:.4f}")
        print("Aggregated Feature Importances (head):\n",
              self.aggregated_importances.to_string(index=False))
        print("Aggregated Confusion Matrix:\n", total_conf_matrix)

        plt.figure(figsize=(6, 4))
        sns.heatmap(total_conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Reversal", "Reversal"],
                    yticklabels=["No Reversal", "Reversal"])
        plt.title("Aggregated Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    def get_predictions(self):
        # series_list = [
        #     pd.Series(self.results[fold]["predictions"])
        #     for fold in self.results:
            

        # ]
        all_predictions = {}
        for fold in self.results: 
            all_predictions.update(self.results[fold]["predictions_with_dates"])

       # predictions = { key.date(): value for key, value in zip(self.df.index, all_predictions)} 

        return all_predictions
    
    def get_top_features(self, n=25): 
        """
        Returns the top-n features from self.aggregated_importances.
        Should be called AFTER run_model() + get_overall_results() on the full feature set.
        """
        if self.aggregated_importances is None:
            raise ValueError("You must run_model() and get_overall_results() first to populate feature importances.")
        return self.aggregated_importances.head(n)['feature'].tolist()


# rf = RandomForestModel()
# rf.run_model()
# print(rf.get_predictions())
# # print(len(rf.df))
# rf.get_overall_results()
