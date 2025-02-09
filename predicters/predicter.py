from dataloader import DataLoader
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, recall_score
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

class RandomForestModel: 
    def __init__(self, dl=DataLoader(), tune=False):
        self.df = dl.get_data()
        self.predictions={}
        self.tune = tune

    def run_model(self):
        df = self.df
        X = df.drop(columns=["Reversal"])

        #Extra cleaning
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(method='ffill', inplace=True)
        X = X.astype(np.float64)


        y = df["Reversal"].astype(int)

        tscv = TimeSeriesSplit(n_splits=5)

        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            print(f"Fold {fold + 1}")
            print("Training data indices:", train_index)
            print("Test data indices:", test_index)
            print("Training data time period:", X_train.index[0], "to", X_train.index[-1])
            print("Test data timeperiod:", X_test.index[0], "to", X_test.index[-1])
            print("-" * 40)

            if self.tune:
                param_grid = {
                    'max_features': ['sqrt', 'log2', None],
                    #'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
                base_model = RandomForestClassifier(random_state=42)
                grid = GridSearchCV(base_model, param_grid, cv=3, scoring='f1', n_jobs=-1)

                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                print("Bästa parametrar för fold", fold+1, ":", grid.best_params_)
            else:
                # Använd tidigare tunade parametrar för 5 folds
                best_model = RandomForestClassifier(random_state=42, max_depth=10, min_samples_split=2, max_features='sqrt')
                best_model.fit(X_train, y_train)
            
            # Gör prediktioner på testdatan
            predictions = best_model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            conf_matrix = confusion_matrix(y_test, predictions)
            importances = best_model.feature_importances_
            feature_names = X_train.columns
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values(by="importance", ascending=False)


            # Spara resultatet för denna fold
            self.predictions[fold] = {
                "predictions": predictions,
                "accuracy": acc,
                "f1": f1,
                "recall": recall,
                "confusion_matrix": conf_matrix,
                "model": best_model,
                "feature_importances": importance_df
            }

            print(f"Fold {fold + 1} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}\n")

        #self.get_overall_results()
        
    def get_overall_results(self):
        
        overall_acc = sum([self.predictions[fold]["accuracy"] for fold in self.predictions]) / len(self.predictions)
        overall_f1 = sum([self.predictions[fold]["f1"] for fold in self.predictions]) / len(self.predictions)
        overall_recall = sum([self.predictions[fold]["recall"] for fold in self.predictions]) / len(self.predictions)
        total_conf_matrix = np.zeros_like(list(self.predictions.values())[0]["confusion_matrix"])

        for fold in self.predictions:
            total_conf_matrix += self.predictions[fold]["confusion_matrix"]

        all_importances = [self.predictions[fold]["feature_importances"] for fold in self.predictions]
        importances_concat = pd.concat(all_importances)
        aggregated_importances = importances_concat.groupby('feature', as_index=False)['importance'].mean()
        aggregated_importances = aggregated_importances.sort_values(by="importance", ascending=False)
        
        print(f"Overall accuracy: {overall_acc:.4f}")
        print(f"Overall F1-score: {overall_f1:.4f}")
        print(f"Overall Recall: {overall_recall:.4f}")
        print(f"Overall feature importances:\n", aggregated_importances)
        print("Aggregated Confusion Matrix:\n", total_conf_matrix)

        plt.figure(figsize=(6, 4))
        sns.heatmap(total_conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title("Aggregated Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    def get_predictions(self):
        series_list = [
            pd.Series(self.predictions[fold]["predictions"])
            for fold in self.predictions
        ]
        all_predictions = pd.concat(series_list, axis=0, ignore_index=True)
        return all_predictions

# rf = RandomForestModel()
# rf.run_model()
# print(rf.get_predictions())
# print(len(rf.df))