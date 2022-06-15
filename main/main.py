import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def get_integrating_order(data: pd.DataFrame, column: str) -> int:
    data_copy = data.copy(deep=True)
    for i in range(1, 20):
        adf_result = adfuller(data_copy[column])
        if adf_result[1] < 0.05:
            return i
        else:
            data_copy = data_copy.diff().dropna()
    return -1


def get_regression_order(data: pd.DataFrame, column: str) -> int:
    pacf_results = pacf(data[column])
    return np.argmax(pacf_results[1:]) + 1


def get_ma_order(data: pd.DataFrame, column: str) -> int:
    acf_results = acf(data[column])
    return np.argmax(acf_results[1:]) + 1


def main():
    df = pd.read_csv('../data/MANA-USD.csv', parse_dates=['Date'], index_col=['Date'])
    print(df.tail())
    print(df.columns)

    train_start = '2022-03-10'
    train_end = '2022-04-20'
    test_start = '2022-04-21'
    test_end = '2022-05-03'

    whole = df.loc[train_start:test_end]
    train = df.loc[train_start:train_end]
    test = df.loc[test_start:test_end]
    whole = whole.iloc[:, 1:2]
    train = train.iloc[:, 1:2]
    test = test.iloc[:, 1:2]
    print(train)
    print(test)

    p = get_regression_order(train, 'High')
    d = get_integrating_order(train, 'High')
    q = get_ma_order(train, 'High')

    if d == -1:
        print('couldnt difference')
        return

    arima_model = ARIMA(train, order=(p, d, q))
    model = arima_model.fit()
    print(model.summary())

    prediction = model.predict(start=test_start, end=test_end)
    whole.shift().plot()
    model.fittedvalues[1:].plot()
    prediction.plot(c='orange')
    plt.show()


if __name__ == '__main__':
    main()