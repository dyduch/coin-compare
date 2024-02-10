import math
import numpy as np
import pandas as pd
from matplotlib.dates import date2num
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

class SVRMethod:
    def __init__(self, kernel: str = 'rbf',
                 c_reg: float = 1.0,
                 gamma='scale', epsilon: float = 0.1):
        self.kernel = kernel
        self.c_reg = c_reg
        self.gamma = gamma
        self.epsilon = epsilon
        self.scaler = MinMaxScaler()

    def fit(self, data: pd.DataFrame, column: str, split_index: int) -> SVR:

        filtered_data = data.loc[:, [column]]
        scaled_data = pd.DataFrame(self.scaler.fit_transform(filtered_data), columns=filtered_data.columns, index=filtered_data.index)

        train_df = scaled_data.iloc[:split_index, :]
        train_df = train_df.reset_index()
        train_df['Date'] = train_df['Date'].map(date2num)

        dates = train_df['Date'].values
        prices = train_df[column].values

        dates = np.reshape(dates, (len(dates), 1))

        model = SVR(kernel=self.kernel, C=self.c_reg, gamma=self.gamma, epsilon=self.epsilon)
        model.fit(dates, prices)
        return model


    def predict(self, model: SVR, test_df: pd.DataFrame, column: str) -> pd.Series:
        filtered_data = test_df.loc[:, [column]]
        df = filtered_data.reset_index()
        original_dates = df['Date'].values
        df['Date'] = df['Date'].map(date2num)

        dates = df['Date'].values
        dates = np.reshape(dates, (len(dates), 1))

        result = model.predict(dates)

        result = self.scaler.inverse_transform(result.reshape(-1, 1))
        result = result.reshape(1, -1).flatten()
        result = pd.Series(result, index=original_dates)
        return result
