import math
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_squared_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', ValueWarning)
warnings.simplefilter('ignore', UserWarning)


class ArimaMethod:
    def __init__(self, compute_parameters: bool = False,
                 max_integrating_order: int = 20,
                 max_p: int = 5, max_d: int = 5, max_q: int = 5):
        self.compute_parameters = compute_parameters
        self.max_integrating_order = max_integrating_order
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q

    def fit(self, data: pd.DataFrame, column: str, split_index: int) -> ARIMAResults:
        filtered_data = data.loc[:, [column]]
        train_df = filtered_data.iloc[:split_index, :]
        test_df = filtered_data.iloc[split_index:, :]

        if self.compute_parameters:
            return self.compute_model(train_df, test_df, column)
        else:
            p, d, q = self.get_arima_parameters(train_df, column)
            arima_model = ARIMA(train_df, order=(p, d, q))
            return arima_model.fit()

    def predict(self, model: ARIMA, test_df: pd.DataFrame, column: str) -> pd.Series:
        df = test_df.loc[:, [column]]
        start, end = self.get_start_end(df)
        return model.predict(start, end)

    def compute_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame, column: str) -> ARIMAResults:
        best_p, best_d, best_q = self.get_arima_parameters(train_df, column)
        best_model, best_rmse = self.compute_model_and_rmse(train_df, test_df, best_p, best_d, best_q)

        for p in range(self.max_p):
            for d in range(self.max_d):
                for q in range(self.max_q):
                    try:
                        new_model, new_rmse = self.compute_model_and_rmse(train_df, test_df, p, d, q)
                        print("New rmse: {0} for parameters (p={1}, d={2}, q={3}) - previous best rmse: {4}".format(
                            new_rmse, p, d, q, best_rmse))
                        if new_rmse < best_rmse:
                            best_model = new_model
                            best_rmse = new_rmse
                    except:
                        print('Parameters: (p={0}, d={1}, q={2}) caused error'.format(p, d, q))
                        continue
        return best_model

    def get_arima_parameters(self, data: pd.DataFrame, column: str) -> (int, int, int):
        return (
            self.get_regression_order(data, column),
            self.get_integrating_order(data, column),
            self.get_moving_avg_order(data, column)
        )

    def get_integrating_order(self, data: pd.DataFrame, column: str) -> int:
        data_copy = data.copy(deep=True)
        for i in range(1, self.max_integrating_order):
            adf_result = adfuller(data_copy[column])
            if adf_result[1] < 0.05:
                return i
            else:
                data_copy = data_copy.diff().dropna()
        return -1

    def get_regression_order(self, data: pd.DataFrame, column: str) -> int:
        pacf_results = pacf(data[column])
        return np.argmax(pacf_results[1:]) + 1

    def get_moving_avg_order(self, data: pd.DataFrame, column: str) -> int:
        acf_results = acf(data[column])
        return np.argmax(acf_results[1:]) + 1

    def compute_model_and_rmse(self, train_df: pd.DataFrame, test_df: pd.DataFrame, p: int, d: int, q: int) -> (
            pd.DataFrame, float):
        test_start, test_end = self.get_start_end(test_df)

        model = ARIMA(train_df, order=(p, d, q))
        model = model.fit()
        prediction = model.predict(start=test_start, end=test_end)
        rmse = mean_squared_error(test_df, prediction, squared=False)
        return model, rmse

    def get_start_end(self, data: pd.DataFrame) -> (str, str):
        test_start = data.iloc[0].name
        test_end = data.iloc[len(data.values) - 1].name
        return test_start, test_end
