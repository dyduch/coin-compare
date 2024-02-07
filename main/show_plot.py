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

class ModelData:
    name: str
    model_params: List[NamedModelParameter]

    def __init__(self, name: str, model_params: List[NamedModelParameter]):
        self.name = name
        self.model_params = model_params

class PriceData:
    dates: ndarray
    prices: ndarray

    def __init__(self, dates: ndarray, prices: ndarray):
        self.dates = dates
        self.prices = prices


class PredictionPriceData(PriceData):
    model_data: ModelData
    color: str

    def __init__(self, dates: ndarray, prices: ndarray, model_data: ModelData, color: str):
        super().__init__(dates, prices)
        self.model_data = model_data
        self.color = color


def show_plot(currency: str, actual_price_data: PriceData, prediction_data: List[PredictionPriceData]):
    title = get_title(currency)
    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes()
    plt.plot(actual_price_data.dates, actual_price_data.prices, color='black', linestyle='-', label='Cena')
    for prediction in prediction_data:
        plt.plot(prediction.dates, prediction.prices,
                 color=prediction.color, label=get_prediction_label(prediction.model_data))
    ax.xaxis.set_major_locator(DayLocator(interval=14))
    plt.xlabel('Data')
    plt.ylabel('Cena w USD')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def get_title(currency: str) -> str:
    title = ('Wykres faktycznych cen {0}'
             ' i przewidzianych wartości przez model').format(currency)
    return title

def get_prediction_label(model_data: ModelData) -> str:
    label = '{0} ('.format(model_data.name)
    for param in model_data.model_params:
        label += '{0}: {1}, '.format(param.name, param.value)
    label = label[:-2]
    label += ')'
    return label