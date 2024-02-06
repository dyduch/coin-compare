from datetime import datetime, timedelta
from typing import List, Tuple

import pandas
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()


def get_data(start: datetime, end: datetime, currency: str, test_size: int = 0) -> pandas.DataFrame:
    end = end + timedelta(days=test_size)
    df = pdr.get_data_yahoo([currency], start, end)
    df = df.sort_values('Date')
    df = df.resample('D').bfill()
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)

    return df


def single_set_test() -> Tuple[datetime, datetime]:
    return datetime(2023, 6, 1), datetime(2023, 11, 1)


def variable_train_set_test() -> List[Tuple[datetime, datetime]]:
    return [
        (datetime(2022, 4, 1), datetime(2022, 7, 1)),
        (datetime(2021, 8, 1), datetime(2021, 10, 31)),
        (datetime(2020, 7, 1), datetime(2020, 9, 30))
    ]


def variable_train_set_test_single(idx: int) -> Tuple[datetime, datetime]:
    return variable_train_set_test()[idx]


def variable_train_length_test() -> List[Tuple[datetime, datetime]]:
    return [
        (datetime(2022, 4, 1), datetime(2022, 7, 1)),
        (datetime(2022, 1, 1), datetime(2022, 7, 1)),
        (datetime(2021, 6, 1), datetime(2022, 7, 1)),
        (datetime(2020, 1, 1), datetime(2022, 7, 1)),
        (datetime(2018, 1, 1), datetime(2022, 7, 1))
    ]


def variable_train_length_test_single(idx: int) -> Tuple[datetime, datetime]:
    return variable_train_length_test()[idx]
