import math
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import layers


class LSTMMethod:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.scaler = MinMaxScaler()

    def fit(self, data: pd.DataFrame, column: str, split: float) -> SVR:
        filtered_data = data.loc[:, [column]]
        split_index = math.ceil(len(filtered_data.values) * split)

        train_df = filtered_data.iloc[:split_index, :]

        prices = train_df[column].values

        scaled_prices = self.scaler.fit_transform(prices.reshape(-1, 1))

        x_train, y_train = self.prepare_training_sets(scaled_prices)

        model = keras.Sequential()
        model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(layers.LSTM(100, return_sequences=False))
        model.add(layers.Dense(25))
        model.add(layers.Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=3)

        return model

    def predict(self, model: keras.Sequential, data: pd.DataFrame, column: str, split: float) -> pd.Series:
        filtered_data = data.loc[:, [column]]
        split_index = math.ceil(len(filtered_data.values) * split)
        prices = filtered_data[column].values

        test_df = data.iloc[split_index:, :]
        df = test_df.reset_index()
        original_dates = df['Date'].values

        x_test, y_test = self.prepare_test_sets(prices, split_index)

        predictions = model.predict(x_test)
        predictions = self.scaler.inverse_transform(predictions).ravel()
        return pd.Series(predictions, index=original_dates)

    def prepare_training_sets(self, prices):
        x_train, y_train = [], []

        for i in range(self.window_size, len(prices)):
            x_train.append(prices[i - self.window_size: i, 0])
            y_train.append(prices[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train

    def prepare_test_sets(self, prices, split_index):
        scaled_prices = self.scaler.fit_transform(prices.reshape(-1, 1))
        filtered_prices = scaled_prices[split_index - self.window_size:, :]

        x_test, y_test = [], prices[split_index:]

        for i in range(self.window_size, len(filtered_prices)):
            x_test.append(filtered_prices[i - self.window_size:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return x_test, y_test
