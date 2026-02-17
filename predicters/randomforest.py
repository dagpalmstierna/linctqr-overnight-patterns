import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from dataloader import DataLoader


class RandomForestModel:
    def __init__(self, dl=DataLoader(), tune=False):
        self.dl = dl
        self.df = dl.get_data()
        self.results = {}
        self.tune = tune
        self.aggregated_importances = None

    def run_model(self, top_features=None):
        df = self.df
        X = df.drop(columns=["Reversal"])

        if top_features is not None:
            X = X[top_features]

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(method='ffill', inplace=True)
        X = X.astype(np.float64)

        y = df["Reversal"].astype(int)
        tscv = TimeSeriesSplit(n_splits=5)

        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]

            print(f"\nFold {fold + 1}")
            print(f"  Train: {train_df.index[0]} to {train_df.index[-1]}")
            print(f"  Test:  {test_df.index[0]} to {test_df.index[-1]}")

            # Balance training data with time-aware SMOTE (test data left untouched)
            train_df_balanced = self.dl.balance_training_data_with_time(train_df)

            X_train = train_df_balanced.drop(columns=["Reversal"])
            y_train = train_df_balanced["Reversal"].astype(int)
            X_test = test_df.drop(columns=["Reversal"])
            y_test = test_df["Reversal"].astype(int)

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
                print(f"  Best params: {grid.best_params_}")
            else:
                rf_model = RandomForestClassifier(
                    random_state=42, max_depth=10,
                    min_samples_split=2, max_features=None, n_estimators=100
                )
                rf_model.fit(X_train, y_train)

            rf_probs = rf_model.predict_proba(X_test)[:, 1]
            predictions = (rf_probs > 0.5).astype(int)

            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            conf = confusion_matrix(y_test, predictions)

            importances = rf_model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': importances
            }).sort_values(by="importance", ascending=False)

            pred_dates = pd.DatetimeIndex(test_df.index.tz_localize(None))
            fold_pred_dict = dict(zip(pred_dates, predictions))

            self.results[fold] = {
                "predictions": predictions,
                "predictions_with_dates": fold_pred_dict,
                "accuracy": acc,
                "f1": f1,
                "recall": recall,
                "confusion_matrix": conf,
                "model": rf_model,
                "feature_importances": importance_df
            }

            print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")

    def get_overall_results(self):
        overall_acc = np.mean([r["accuracy"] for r in self.results.values()])
        overall_f1 = np.mean([r["f1"] for r in self.results.values()])
        overall_recall = np.mean([r["recall"] for r in self.results.values()])

        total_conf = np.zeros_like(list(self.results.values())[0]["confusion_matrix"])
        for r in self.results.values():
            total_conf += r["confusion_matrix"]

        all_importances = [r["feature_importances"] for r in self.results.values()]
        importances_concat = pd.concat(all_importances)
        self.aggregated_importances = (
            importances_concat.groupby('feature', as_index=False)['importance']
            .mean()
            .sort_values(by="importance", ascending=False)
        )

        print(f"\nOverall Accuracy: {overall_acc:.4f}")
        print(f"Overall F1:       {overall_f1:.4f}")
        print(f"Overall Recall:   {overall_recall:.4f}")
        print("\nAggregated Feature Importances:")
        print(self.aggregated_importances.to_string(index=False))
        print("\nAggregated Confusion Matrix:")
        print(total_conf)

        plt.figure(figsize=(6, 4))
        sns.heatmap(total_conf, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Reversal", "Reversal"],
                    yticklabels=["No Reversal", "Reversal"])
        plt.title("Aggregated Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    def get_predictions(self):
        all_predictions = {}
        for r in self.results.values():
            all_predictions.update(r["predictions_with_dates"])
        return all_predictions

    def get_top_features(self, n=25):
        """Return the top-n most important features (call after run_model + get_overall_results)."""
        if self.aggregated_importances is None:
            raise ValueError("Run run_model() and get_overall_results() first.")
        return self.aggregated_importances.head(n)['feature'].tolist()
