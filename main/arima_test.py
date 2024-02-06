import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from matplotlib.dates import DayLocator, MonthLocator

from arima_method import ArimaMethod
from statsmodels.tsa.arima_model import ARIMAResults

from get_data import get_data, variable_train_set_test_single

def main():
    column_name = 'Close'
    currency = 'BTC-USD'
    start_end_pairs = [variable_train_set_test_single(2)]
    for pair in start_end_pairs:
        rmse_table = []
        mape_table = []
        rmspe_table = []
        print("\n\n")
        print(pair[0], pair[1])
        arima_model = None
        arima_model_comp = None
        for test_sample_size in [1, 7, 21, 60]:
            start, end = pair

            df = get_data(start, end, currency, test_sample_size)

            dates_df = df.copy()
            dates_df = dates_df.reset_index()
            dates = dates_df['Date'].values
            train_dates = dates[:-test_sample_size]
            prices = df[column_name].values

            # print(len(dates))
            # print(dates_df.head(1))
            # print(dates_df.tail(1))

            if arima_model is None:
                arima_model = get_arima_model(df, column_name, test_sample_size, compute_values=False)

            if arima_model_comp is None:
                arima_model_comp = get_arima_model(df, column_name, test_sample_size, compute_values=True)

            arima_results, arima_rmse, arima_mape, arima_rmspe = get_arima_predictions_size(arima_model, df, column_name,
                                                                                            test_sample_size)
            print("RMSE: {0}, MAPE: {1}, RMSPE: {2}, test sample: {2}".format(arima_rmse, arima_mape, arima_rmspe,
                                                                              test_sample_size))

            arima_results_c, arima_rmse_c, arima_mape_c, arima_rmspe_c = get_arima_predictions_size(arima_model_comp, df,
                                                                                            column_name,
                                                                                            test_sample_size)
            print("RMSE: {0}, MAPE: {1}, RMSPE: {2}, test sample: {3}".format(arima_rmse, arima_mape, arima_rmspe,
                                                                              test_sample_size))

            print("Computed: RMSE: {0}, MAPE: {1}, RMSPE: {2}, test sample: {3}".format(arima_rmse_c, arima_mape_c, arima_rmspe_c,
                                                                              test_sample_size))

            print(arima_results_c.values)
            # svr_results = get_svr_predictions(df, column_name, split)
            # lstm_results = get_lstm_predictions(df, column_name, split)
            rmse_table.append(arima_rmse)
            mape_table.append(arima_mape)
            rmspe_table.append(arima_rmspe)
            split_index = len(df.values) - test_sample_size - 1
            test_dates = dates[split_index:]
            fig = plt.figure(figsize=(12, 7))
            ax = plt.axes()
            plt.plot(dates, prices, color='black', linestyle='-', label='Cena')
            plt.plot(train_dates[1:-2], arima_model.fittedvalues[2:], color='lightseagreen',
                     label='Wartości dopasowane przez model ARIMA(1, 2, 1)')
            plt.plot(train_dates[3:-2], arima_model_comp.fittedvalues[4:], color='peru',
                     label='Wartości dopasowane przez model ARIMA(4, 4, 2)')
            plt.plot(test_dates, arima_results_c.values, color='saddlebrown', linestyle='--',
                     label='Wartości przewidziane przez model ARIMA(4, 4, 2)')
            plt.plot(test_dates, arima_results.values, color='mediumorchid', linestyle='--',
                     label='Wartości przewidziane przez model ARIMA(1, 2, 1)')
            # plt.plot(test_dates, lstm_results.values, color='green', label='LSTM')
            ax.xaxis.set_major_locator(DayLocator(interval=14))
            plt.xlabel('Data')
            plt.ylabel('Cena w USD')
            plt.title('Wykres cen {0} wraz z dopasowanym modelem ARIMA i przewidzianymi wartościami'.format(currency))
            plt.legend()
            plt.grid()
            plt.show()

        print(" & ".join(map(str, rmse_table)))
        print(" & ".join(map(str, mape_table)))
        print(" & ".join(map(str, rmspe_table)))

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