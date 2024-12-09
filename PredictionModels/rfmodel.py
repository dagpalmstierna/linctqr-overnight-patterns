from abc import ABC
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_recall_curve

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from predicter import PredictionModel
import pandas as pd

class RFModel(PredictionModel):
    def __init__(self) -> None:
        super().__init__()
    
    def perform_tests(self):
        y = self.df["Reversal"]
        top_features = self.importance()["Feature"].values

        X = self.df[top_features]

        column_names = X.columns.tolist()
      
        #Test size 20 % 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_space = {
            'n_estimators': (50, 300),               # Number of trees
            'max_depth': (1, 5),                    # Depth of trees
            'min_samples_split': (20, 35),           # Minimum samples to split a node
            'min_samples_leaf': (3, 8),             # Minimum samples per leaf
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

        #Visualize results
        self.visualize_feature_importance(feature_importance)
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_precision_recall_curve(opt.best_estimator_, X_test, y_test)
        
    def visualize_feature_importance(self, feature_importance):
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'])
        plt.gca().invert_yaxis()  # Flip the y-axis for better readability
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.show()


    def plot_confusion_matrix(self, y_test, y_pred):
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()
   
    def plot_precision_recall_curve(self, best_estimator, X_test, Y_test):
        y_probs = best_estimator.predict_proba(X_test)[:, 1]  # Probabilities for class 1
        precision, recall, thresholds = precision_recall_curve(Y_test, y_probs)

        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()
    
    # Returns the n most important features
    def importance(self, n=10):
        df = self.df
        y = df["Reversal"]
        X = df.drop("Reversal", axis=1)
        
        random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        random_forest.fit(X, y)


        feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': random_forest.feature_importances_})

        feature_df = feature_df.sort_values(by="Importance", ascending=False)
        
        return feature_df.head(n)


pm = RFModel()
pm.clean_data()
pm.perform_tests()



 
