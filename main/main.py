import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

from arima_method import ArimaMethod
from statsmodels.tsa.arima_model import ARIMAResults

from show_plot import show_plot, NamedModelParameter, PriceData, PredictionPriceData, ModelData
from svr_method import SVRMethod
from lstm_method import LSTMMethod

from get_data import get_data, single_set_test


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

                    svr_results, svr_rmse, svr_mape, svr_rmspe = get_svr_predictions(svr_method, svr_model, df,
                                                                                     column_name,
                                                                                     test_sample_size)
                    print("RMSE: {0}, MAPE: {1}, RMSPE: {2}, test sample: {3}".format(svr_rmse, svr_mape, svr_rmspe,
                                                                                      test_sample_size))

                    # lstm_results = get_lstm_predictions(df, column_name, split)
                    rmse_table.append(svr_rmse)
                    mape_table.append(svr_mape)
                    rmspe_table.append(svr_rmspe)
                    split_index = len(df.values) - test_sample_size - 1
                    test_dates = dates[split_index:]

                    show_plot(currency, PriceData(dates, prices),
                              [PredictionPriceData(test_dates, svr_results.values, ModelData('SVR',
                                                                                             [NamedModelParameter('c', c),
                                                                                              NamedModelParameter(
                                                                                                  'gamma', gamma),
                                                                                              NamedModelParameter(
                                                                                                  'epsilon', svr_eps)]),
                                                   'teal')])

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
