import math

import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from arima_method import ArimaMethod
from svr_method import SVRMethod
from lstm_method import LSTMMethod


def main():
    column_name = 'Close'



    start_end_pairs = [
        (datetime.datetime(2022, 4, 1), datetime.datetime(2022, 7, 1)),
        (datetime.datetime(2022, 1, 1), datetime.datetime(2022, 7, 1)),
        (datetime.datetime(2021, 6, 1), datetime.datetime(2022, 7, 1)),
        (datetime.datetime(2020, 1, 1), datetime.datetime(2022, 7, 1)),
        (datetime.datetime(2018, 1, 1), datetime.datetime(2022, 7, 1))
    ]

    for pair in start_end_pairs:
        rmse_table = []
        mape_table = []
        rmspe_table = []
        print("\n\n")
        print(pair[0], pair[1])
        for test_sample_size in [1, 7, 21, 60]:
            start, end = pair

            df = get_data(start, end, test_sample_size)

            dates_df = df.copy()
            dates_df = dates_df.reset_index()
            dates = dates_df['Date'].values
            prices = df[column_name].values

            print(len(dates))
            # print(dates_df.head(1))
            # print(dates_df.tail(1))

            arima_results, arima_rmse, arima_mape, arima_rmspe = get_arima_predictions_size(df, column_name, test_sample_size)
            print("RMSE: {0}, MAPE: {1}, RMSPE: {2}, test sample: {2}".format(arima_rmse, arima_mape, arima_rmspe, test_sample_size))
            # svr_results = get_svr_predictions(df, column_name, split)
            # lstm_results = get_lstm_predictions(df, column_name, split)
            rmse_table.append(arima_rmse)
            mape_table.append(arima_mape)
            rmspe_table.append(arima_rmspe)
            split_index = len(df.values) - test_sample_size
            test_dates = dates[split_index:]

            plt.figure(figsize=(12, 6))
            plt.plot(dates, prices, color='black', label='Prices')
            plt.plot(test_dates, arima_results.values, color='red', label='Arima')
            # plt.plot(test_dates, svr_results.values, color='orange', label='SVR')
            # plt.plot(test_dates, lstm_results.values, color='green', label='LSTM')

            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()

        print(" & ".join(map(str, rmse_table)))
        print(" & ".join(map(str, mape_table)))
        print(" & ".join(map(str, rmspe_table)))


def get_data(start, end, test_size: int = 0):
    end = end + datetime.timedelta(days=test_size)
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


def get_arima_predictions_split(df: pd.DataFrame, column_name: str, split: float):
    split_index = math.ceil(len(df.values) * split)
    return get_arima_predictions(column_name, df, split_index)


def get_arima_predictions_size(df: pd.DataFrame, column_name: str, test_size_saple: int):
    split_index = len(df.values) - test_size_saple
    return get_arima_predictions(column_name, df, split_index)


def get_arima_predictions(column_name, df, split_index):
    arima = ArimaMethod(compute_parameters=False)
    test_df = df.iloc[split_index:, :]
    test_df = test_df.loc[:, [column_name]]
    arima_result = arima.fit(df, column_name, split_index)
    # print(arima_result.summary())
    prediction = arima.predict(arima_result, test_df, column_name)
    rmse = mean_squared_error(test_df, prediction, squared=False)
    mape = mean_absolute_percentage_error(test_df, prediction)
    rmpse = np.sqrt(np.mean(np.square(((test_df.values - prediction.values) / test_df.values))))
    return prediction, round(rmse, 3), str(round(100 * mape, 3)) + "\%", str(round(100 * rmpse, 3)) + "\%"


if __name__ == '__main__':
    main()
