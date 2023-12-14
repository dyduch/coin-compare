import math

import numpy as np
import pandas as pd
import datetime
from pandas_datareader import data as pdr
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from matplotlib.dates import DayLocator, MonthLocator
from sklearn.preprocessing import MinMaxScaler

from arima_method import ArimaMethod
from statsmodels.tsa.arima_model import ARIMAResults
from svr_method import SVRMethod
from lstm_method import LSTMMethod

yf.pdr_override()

def main():
    column_name = 'Close'
    currency = 'BTC-USD'
    start_end_pairs = [single_set_test()]
    top_3_rmse = []
    svr_gammas = [0.008, 0.010, 0.012, 'scale']
    svr_cs = [256, 60, 45]
    svr_eps = 0.05
    for gamma in svr_gammas:
        for c in svr_cs:
            print("SVR: c={0}, gamma={1}".format(c, gamma))
            svr_method = SVRMethod(c_reg=c, gamma=gamma, epsilon=svr_eps)
            for pair in start_end_pairs:
                rmse_table = []
                mape_table = []
                rmspe_table = []
                print("\n\n")
                print(pair[0], pair[1])
                svr_model = None
                for test_sample_size in [30]:
                    start, end = pair

                    df = get_data(start, end, currency, test_sample_size)

                    scaler = MinMaxScaler()
                    arr_scaled = scaler.fit_transform(df)

                    df = pd.DataFrame(arr_scaled, columns=df.columns, index=df.index)

                    dates_df = df.copy()
                    dates_df = dates_df.reset_index()
                    dates = dates_df['Date'].values
                    train_dates = dates[:-test_sample_size]
                    prices = df[column_name].values

                    # print(len(dates))
                    # print(dates_df.head(1))
                    # print(dates_df.tail(1))

                    if svr_model is None:
                        svr_model = get_svr_model(svr_method, df, column_name, test_sample_size)

                    svr_results, svr_rmse, svr_mape, svr_rmspe = get_svr_predictions(svr_method, svr_model, df, column_name,
                                                                                                    test_sample_size)
                    print("RMSE: {0}, MAPE: {1}, RMSPE: {2}, test sample: {3}".format(svr_rmse, svr_mape, svr_rmspe,
                                                                                      test_sample_size))

                    # lstm_results = get_lstm_predictions(df, column_name, split)
                    rmse_table.append(svr_rmse)
                    mape_table.append(svr_mape)
                    rmspe_table.append(svr_rmspe)
                    split_index = len(df.values) - test_sample_size - 1
                    test_dates = dates[split_index:]

                    fig = plt.figure(figsize=(12, 7))
                    ax = plt.axes()
                    plt.plot(dates, prices, color='black', linestyle='-', label='Cena')
                    plt.plot(test_dates, svr_results.values, color='teal', label='SVR')
                    ax.xaxis.set_major_locator(DayLocator(interval=14))
                    plt.xlabel('Data')
                    plt.ylabel('Cena w USD')
                    plt.title(
                        'Wykres cen {0} wraz z dopasowanym modelem SVR(c={1}, gamma={2}, eps={3} i przewidzianymi wartościami'.format(currency, c, gamma, svr_eps))
                    plt.legend()
                    plt.grid()
                    plt.show()

                # if len(top_3_rmse) < 3:
                #     top_3_rmse.append((rmse_table, c, gamma))
                # else:
                #     for i in range(len(top_3_rmse)):
                #         if top_3_rmse[i][0][3] > rmse_table[3]:
                #             top_3_rmse[i] = (rmse_table, c, gamma)
                #             break
                #
                #
                # print(" & ".join(map(str, rmse_table)))
                # print(" & ".join(map(str, mape_table)))
                # print(" & ".join(map(str, rmspe_table)))
    print(top_3_rmse)



def single_set_test():
    return datetime.datetime(2023, 6, 1), datetime.datetime(2023, 11, 1)


def variable_train_set_test():
    return [
        (datetime.datetime(2022, 4, 1), datetime.datetime(2022, 7, 1)),
        (datetime.datetime(2021, 8, 1), datetime.datetime(2021, 10, 31)),
        (datetime.datetime(2020, 7, 1), datetime.datetime(2020, 9, 30))
    ]

def variable_train_set_test_single(idx):
    return variable_train_set_test()[idx]


def variable_train_length_test():
    return [
        (datetime.datetime(2022, 4, 1), datetime.datetime(2022, 7, 1)),
        (datetime.datetime(2022, 1, 1), datetime.datetime(2022, 7, 1)),
        (datetime.datetime(2021, 6, 1), datetime.datetime(2022, 7, 1)),
        (datetime.datetime(2020, 1, 1), datetime.datetime(2022, 7, 1)),
        (datetime.datetime(2018, 1, 1), datetime.datetime(2022, 7, 1))
    ]

def variable_train_length_test_single(idx):
    return variable_train_length_test()[idx]


def get_data(start, end, currency, test_size: int = 0):
    end = end + datetime.timedelta(days=test_size)
    df = pdr.get_data_yahoo([currency], start, end)
    df = df.sort_values('Date')
    df = df.resample('D').bfill()
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)

    return df


def get_lstm_predictions(df: pd.DataFrame, column_name: str, split: float):
    lstm = LSTMMethod()

    model = lstm.fit(df, column_name, split)
    return lstm.predict(model, df, column_name, split)


def get_svr_model(svr_method, df: pd.DataFrame, column_name: str, test_sample_size: int):
    split_index = len(df.values) - test_sample_size - 1
    return svr_method.fit(df, column_name, split_index)

def get_svr_predictions(svr_method, svr_model, df: pd.DataFrame, column_name: str, test_sample_size: int):
    split_index = len(df.values) - test_sample_size - 1
    test_df = df.iloc[split_index:, :]
    test_df = test_df.loc[:, [column_name]]

    prediction = svr_method.predict(svr_model, test_df, column_name)
    rmse = mean_squared_error(test_df, prediction, squared=False)
    mape = mean_absolute_percentage_error(test_df, prediction)
    rmpse = np.sqrt(np.mean(np.square(((test_df.values - prediction.values) / test_df.values))))
    return prediction, round(rmse, 3), str(round(100 * mape, 3)) + "\%", str(round(100 * rmpse, 3)) + "\%"


def get_arima_model(df: pd.DataFrame, column_name: str, test_sample_size: int, compute_values):
    arima = ArimaMethod(compute_parameters=compute_values)
    split_index = len(df.values) - test_sample_size - 1
    return arima.fit(df, column_name, split_index)


def get_arima_predictions_split(model: ARIMAResults, df: pd.DataFrame, column_name: str, split: float):
    split_index = math.ceil(len(df.values) * split)
    return get_arima_predictions(model, df, column_name, split_index)


def get_arima_predictions_size(model: ARIMAResults, df: pd.DataFrame, column_name: str, test_size_saple: int):
    split_index = len(df.values) - test_size_saple - 1
    return get_arima_predictions(model, df, column_name, split_index)

def get_arima_predictions(model, df, column_name, split_index):
    arima = ArimaMethod(compute_parameters=True)
    test_df = df.iloc[split_index:, :]
    test_df = test_df.loc[:, [column_name]]
    # print(arima_result.summary())
    prediction = arima.predict(model, test_df, column_name)
    rmse = mean_squared_error(test_df, prediction, squared=False)
    mape = mean_absolute_percentage_error(test_df, prediction)
    rmpse = np.sqrt(np.mean(np.square(((test_df.values - prediction.values) / test_df.values))))
    return prediction, round(rmse, 3), str(round(100 * mape, 3)) + "\%", str(round(100 * rmpse, 3)) + "\%"


if __name__ == '__main__':
    main()
