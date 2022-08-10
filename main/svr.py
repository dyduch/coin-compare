import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import pandas_datareader.data as web
import datetime
import matplotlib.dates as mdates

register_matplotlib_converters()

# https://towardsdatascience.com/walking-through-support-vector-regression-and-lstms-with-stock-price-prediction-45e11b620650

def main():
    start = datetime.datetime(2020, 1, 1)
    end = datetime.datetime(2022, 6, 30)
    df = web.DataReader('BTC-USD', 'yahoo', start, end)

    test_start = datetime.datetime(2022, 7, 1)
    test_end = datetime.datetime.today()
    test_df = web.DataReader('BTC-USD', 'yahoo', test_start, test_end)

    df = df.sort_values('Date')
    test_df = test_df.sort_values('Date')

    # fix the date
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    test_df.reset_index(inplace=True)
    test_df.set_index("Date", inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(df["High"])
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('High Price', fontsize=15)

    # Rolling mean
    close_px = df['High']
    mavg = close_px.rolling(window=100).mean()
    plt.figure(figsize=(12, 6))
    close_px.plot(label='BTC-USD')
    mavg.plot(label='mavg')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # change the dates into ints for training
    dates_df = df.copy()
    dates_df = dates_df.reset_index()

    dates_test_df = test_df.copy()
    dates_test_df = dates_test_df.reset_index()

    # Store the original dates for plotting the predicitons
    org_dates = dates_df['Date']
    test_org_dates = dates_test_df['Date']

    # convert to ints
    dates_df['Date'] = dates_df['Date'].map(mdates.date2num)
    dates_test_df['Date'] = dates_test_df['Date'].map(mdates.date2num)

    dates = dates_df['Date'].values
    prices = df['High'].values

    test_dates = dates_test_df['Date'].values
    test_prices = dates_test_df['High'].values

    # Convert to 1d Vector
    dates = np.reshape(dates, (len(dates), 1))
    prices = np.reshape(prices, (len(prices), 1)).ravel()

    test_dates = np.reshape(test_dates, (len(test_dates), 1))
    test_prices = np.reshape(test_prices, (len(test_prices), 1)).ravel()

    svr_rbf = SVR(kernel='rbf', C=1e4, gamma=0.0003)
    svr_rbf.fit(dates, prices)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, color='black', label='Data')
    plt.plot(org_dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(test_dates, test_prices, color='brown', label='Data Test')
    plt.plot(test_org_dates, svr_rbf.predict(test_dates), color='green', label='RBF predict')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
