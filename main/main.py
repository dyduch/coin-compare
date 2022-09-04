import math

import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web
from arima_method import ArimaMethod
from svr_method import SVRMethod

def main():
    svr = SVRMethod()

    start = datetime.datetime(2022, 1, 1)
    end = datetime.datetime(2022, 3, 30)
    df = web.DataReader('BTC-USD', 'yahoo', start, end)
    df = df.sort_values('Date')
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    split = 0.95
    split_index = math.ceil(len(df.values) * split)
    train_df = df.iloc[:split_index, :]
    test_df = df.iloc[split_index:, :]

    model = svr.fit(df, 'Close', 0.95)
    result = svr.predict(model, test_df, 'Close')
    print(result)
    print(type(result))

def get_arima_predictions():
    arima = ArimaMethod(compute_parameters=True)
    start = datetime.datetime(2022, 1, 1)
    end = datetime.datetime(2022, 3, 30)
    df = web.DataReader('BTC-USD', 'yahoo', start, end)
    df = df.sort_values('Date')
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    split = 0.95
    split_index = math.ceil(len(df.values) * split)
    train_df = df.iloc[:split_index, :]
    test_df = df.iloc[split_index:, :]
    arima_result = arima.fit(df, 'Close', split)
    print(arima_result.summary())
    result = arima.predict(arima_result, test_df, 'close')


if __name__ == '__main__':
    main()