from typing import List

from numpy import ndarray
from matplotlib import pyplot as plt
from matplotlib.dates import DayLocator


class NamedModelParameter:
    name: str
    value: str | float

    def __init__(self, name: str, value: str | float):
        self.name = name
        self.value = value


def show_plot(currency: str, actual_dates: ndarray, actual_prices: ndarray, test_dates: ndarray,
              predicted_prices: ndarray, model_name: str, model_params: List[NamedModelParameter]):
    title = get_title(currency, model_name, model_params)
    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes()
    plt.plot(actual_dates, actual_prices, color='black', linestyle='-', label='Cena')
    plt.plot(test_dates, predicted_prices, color='teal', label='SVR')
    ax.xaxis.set_major_locator(DayLocator(interval=14))
    plt.xlabel('Data')
    plt.ylabel('Cena w USD')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def get_title(currency: str, model_name: str, model_params: List[NamedModelParameter]) -> str:
    title = ('Wykres faktycznych cen {0}'
             ' i przewidzianych wartości przez {1}\nParametry modelu: ').format(currency, model_name)
    for param in model_params:
        title += '{0}: {1} '.format(param.name, param.value)
    return title
