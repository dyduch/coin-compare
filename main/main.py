import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import mean_squared_error

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
    filenames = ['BTC-USD']
    train_start_dates = ['2022-05-15']
    train_end_dates = ['2022-07-11']
    test_end_dates = ['2022-07-13']

    for file in filenames:
        for train_start in train_start_dates:
            for train_end in train_end_dates:
                for test_end in test_end_dates:
                    run_arima_for_dates(file, train_start, train_end, train_end, test_end)


def run_arima_for_dates(plot_name, train_start, train_end, test_start, test_end):
    file = '../data/{0}.csv'.format(plot_name)
    df = pd.read_csv(file, parse_dates=['Date'], index_col=['Date'])

    whole = df.loc[train_start:test_end]
    train = df.loc[train_start:train_end]
    test = df.loc[test_start:test_end]

    whole = whole.iloc[:, 1:2]
    train = train.iloc[:, 1:2]
    test = test.iloc[:, 1:2]

    best_p = get_regression_order(train, 'High')
    best_d = get_integrating_order(train, 'High')
    best_q = get_ma_order(train, 'High')

    arima_model = ARIMA(train, order=(best_p, best_d, best_q))
    model = arima_model.fit()
    best_model = model
    print(model.summary())

    prediction = model.predict(start=test_start, end=test_end)
    best_prediction = prediction
    plot_result(best_prediction, model, plot_name + ' calculated', test_start, whole)

    best_rmse = mean_squared_error(test, prediction, squared=False)
    print(best_rmse)


    for p in range(0, 5):
        for d in range(0, 5):
            for q in range(0, 5):
                try:
                    new_arima_model = ARIMA(train, order=(p, d, q))
                    new_model = new_arima_model.fit()

                    new_prediction = new_model.predict(start=test_start, end=test_end)
                    new_rmse = mean_squared_error(test, new_prediction, squared=False)
                    if new_rmse < best_rmse:
                        best_model = new_model
                        best_prediction = new_prediction
                        best_rmse = new_rmse
                except:
                    print('Parameters: ({0},{1}, {2}) caused error'.format(p, d, q))
                    continue

    print(best_model.summary())
    plot_result(best_prediction, best_model, plot_name + ' automated', test_start, whole)
    print(best_rmse)

    future_prediction = best_model.predict(start=test_end, end='2022-08-31')
    plot_result(best_prediction, best_model, plot_name + ' automated', test_start, whole, future_prediction=future_prediction)


def plot_result(best_prediction, model, plot_name, test_start, whole, future_prediction=None):
    whole.shift().plot(label='Real data')
    model.fittedvalues[2:].plot(label='Model fitted values')
    best_prediction.plot(c='red', title=plot_name, label='Model predicted values')
    legend = ['Real data', 'Model fitted values', 'Model prediction']
    if future_prediction is not None:
        future_prediction.plot(c='green', label='Future predicted values')
        legend.append('Future predicted values')
    plt.legend(legend)
    plt.axvline(x=test_start, color='black', linestyle='--')
    plt.show()


if __name__ == '__main__':
    main()
