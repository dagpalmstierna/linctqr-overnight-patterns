import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from dataloader import DataLoader

class LSTMModel:
    def __init__(self, dl=DataLoader(), sequence_length=10):
        self.df = dl.get_data()
        self.sequence_length = sequence_length
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.sequence_length, self.df.shape[1] - 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def create_sequences(self, data, target):
        sequences = []
        targets = []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])
            targets.append(target[i + self.sequence_length])
        return np.array(sequences), np.array(targets)

    def run_model(self):
        df = self.df
        features = df.drop(columns=["Reversal"])
        target = df["Reversal"].astype(int)

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)

        X, y = self.create_sequences(scaled_features, target)

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        print(classification_report(y_test, y_pred))
        print("Accuracy on test set:", accuracy_score(y_test, y_pred))


dl = DataLoader()
lstm_model = LSTMModel(dl)
lstm_model.run_model()