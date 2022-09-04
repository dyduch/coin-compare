import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import pandas_datareader.data as web
import datetime
import matplotlib.dates as mdates
import tensorflow as tf
from tensorflow import keras
from keras import layers

# https://medium.com/the-handbook-of-coding-in-finance/stock-prices-prediction-using-long-short-term-memory-lstm-model-in-python-734dd1ed6827

def lstm():
    start = datetime.datetime(2021, 1, 1)
    end = datetime.datetime(2022, 3, 30)
    df = web.DataReader('BTC-USD', 'yahoo', start, end)

    test_start = datetime.datetime(2022, 4, 1)
    test_end = datetime.datetime(2022, 8, 13)

    pred_end = datetime.datetime.today()

    test_df = web.DataReader('BTC-USD', 'yahoo', test_start, test_end)

    total_df = web.DataReader('BTC-USD', 'yahoo', start, test_end)

    df = df.sort_values('Date')
    test_df = test_df.sort_values('Date')
    total_df = total_df.sort_values('Date')

    # fix the date
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    test_df.reset_index(inplace=True)
    test_df.set_index("Date", inplace=True)
    total_df.reset_index(inplace=True)
    total_df.set_index("Date", inplace=True)

    # change the dates into ints for training
    dates_df = df.copy()
    dates_df = dates_df.reset_index()

    dates_test_df = test_df.copy()
    dates_test_df = dates_test_df.reset_index()

    dates_total_df = total_df.copy()
    dates_total_df = dates_total_df.reset_index()

    # Store the original dates for plotting the predicitons
    org_dates = dates_df['Date']
    test_org_dates = dates_test_df['Date']

    # convert to ints
    dates_df['Date'] = dates_df['Date'].map(mdates.date2num)
    dates_test_df['Date'] = dates_test_df['Date'].map(mdates.date2num)
    dates_total_df['Date'] = dates_total_df['Date'].map(mdates.date2num)

    dates = dates_total_df['Date'].values
    prices = total_df['High'].values

    # test_dates = dates_test_df['Date'].values
    # test_prices = dates_test_df['High'].values
    #
    scaler = MinMaxScaler()
    #
    training_data_len = math.ceil(len(prices) * 0.8)
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
    train_prices = scaled_prices[0: training_data_len, :]

    x_train = []
    y_train = []
    batch_size = 1
    #
    for i in range(batch_size, len(train_prices)):
        x_train.append(train_prices[i - batch_size: i, 0])
        y_train.append(train_prices[i, 0])



    x_train, y_train = np.array(x_train), np.array(y_train)
    print(x_train.shape)
    print(y_train.shape)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    print(x_train.shape)
    print(y_train.shape)

    test_data = scaled_prices[training_data_len - batch_size:, :]
    x_test = []
    y_test = prices[training_data_len:]

    for i in range(batch_size, len(test_data)):
        x_test.append(test_data[i - batch_size:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(layers.LSTM(100, return_sequences=False))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1))
    model.summary()
    #
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=3)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    print(predictions)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, color='black', label='Data')
    plt.plot(dates[training_data_len:], predictions, color='red', label='Data Test')
    # plt.plot(test_org_dates, svr_rbf.predict(test_dates), color='green', label='RBF predict')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    lstm()