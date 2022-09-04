import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web
from arima_method import ArimaMethod

def main():
    arima = ArimaMethod(compute_parameters=True)
    start = datetime.datetime(2022, 1, 1)
    end = datetime.datetime(2022, 3, 30)

    df = web.DataReader('BTC-USD', 'yahoo', start, end)
    df = df.sort_values('Date')
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)

    arima_result = arima.fit(df, 'Close', 0.95)
    print(arima_result.summary())

if __name__ == '__main__':
    main()