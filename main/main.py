import math

import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web
from matplotlib import pyplot as plt

from arima_method import ArimaMethod
from svr_method import SVRMethod
from lstm_method import LSTMMethod


def main():
    df = get_data()

    column_name = 'Close'

    dates_df = df.copy()
    dates_df = dates_df.reset_index()

    dates = dates_df['Date'].values
    prices = df[column_name].values

    split = 0.95
    split_index = math.ceil(len(df.values) * split)

    wider_split = 0.90 * split
    wider_split_index = math.ceil(len(df.values) * wider_split)

    test_dates = dates[split_index:]
    closer_dates = dates[wider_split_index:]
    closer_prices = prices[wider_split_index:]

    arima_results = get_arima_predictions(df, column_name, split)
    svr_results = get_svr_predictions(df, column_name, split)
    lstm_results = get_lstm_predictions(df, column_name, split)

    plt.figure(figsize=(12, 6))
    plt.plot(closer_dates, closer_prices, color='black', label='Prices')
    plt.plot(test_dates, arima_results.values, color='red', label='Arima')
    plt.plot(test_dates, svr_results.values, color='orange', label='SVR')
    plt.plot(test_dates, lstm_results.values, color='green', label='LSTM')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()



def get_data():
    start = datetime.datetime(2019, 1, 1)
    end = datetime.datetime(2022, 8, 15)
    df = web.DataReader('BTC-USD', 'yahoo', start, end)
    df = df.sort_values('Date')
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)

    return df


def get_lstm_predictions(df: pd.DataFrame, column_name: str, split: float):
    lstm = LSTMMethod()

    model = lstm.fit(df, column_name, split)
    return lstm.predict(model, df, column_name, split)


def get_svr_predictions(df: pd.DataFrame, column_name: str, split: float):
    svr = SVRMethod(c_reg=10000, gamma=0.0001)

    split_index = math.ceil(len(df.values) * split)
    test_df = df.iloc[split_index:, :]

    model = svr.fit(df, column_name, 0.95)
    return svr.predict(model, test_df, column_name)


def get_arima_predictions(df: pd.DataFrame, column_name: str, split: float):
    arima = ArimaMethod(compute_parameters=False)

    split_index = math.ceil(len(df.values) * split)
    test_df = df.iloc[split_index:, :]

    arima_result = arima.fit(df, column_name, split)
    return arima.predict(arima_result, test_df, column_name)


if __name__ == '__main__':
    main()
