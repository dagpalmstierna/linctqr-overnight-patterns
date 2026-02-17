from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from dataloader import DataLoader


class SVMModel:
    def __init__(self, dl=DataLoader(), tune=False):
        self.df = dl.get_data()
        self.model = SVC(probability=True, kernel="rbf", gamma=1, C=0.01)
        self.tune = tune

    def run_model(self):
        df = self.df
        X = df.drop(columns=["Reversal"])
        y = df["Reversal"].astype(int)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=5)

        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            print(f"\nFold {fold + 1}")

            if self.tune:
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [1, 0.1, 0.01, 0.001],
                    'kernel': ['rbf', 'linear']
                }
                grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, cv=3, scoring='accuracy')
                grid.fit(X_train, y_train)
                self.model = grid.best_estimator_
                print(f"  Best params: {grid.best_params_}")
            else:
                self.model.fit(X_train, y_train)

            y_probs = self.model.predict_proba(X_test)[:, 1]
            predictions = (y_probs > 0.5).astype(int)

            acc = accuracy_score(y_test, predictions)
            print(f"  Accuracy: {acc:.4f}")
            print(classification_report(y_test, predictions))
