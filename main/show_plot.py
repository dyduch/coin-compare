from typing import List

from numpy import ndarray
from matplotlib import pyplot as plt
from matplotlib.dates import DayLocator, DateFormatter

class NamedModelParameter:
    name: str
    value: str | float

    def __init__(self, name: str, value: str | float):
        self.name = name
        self.value = value

class ModelData:
    name: str
    model_params: List[NamedModelParameter]

    def __init__(self, name: str = 'Cena', model_params=None):
        if model_params is None:
            model_params = []
        self.name = name
        self.model_params = model_params

class PriceData:
    dates: ndarray
    prices: ndarray

    def __init__(self, dates: ndarray, prices: ndarray):
        self.dates = dates
        self.prices = prices


class PlotPriceData(PriceData):
    model_data: ModelData
    color: str
    style: str

    def __init__(self, dates: ndarray, prices: ndarray, model_data: ModelData = ModelData(),
                 color: str = 'black', style: str = '-') -> None:
        super().__init__(dates, prices)
        self.model_data = model_data
        self.color = color
        self.style = style

class Plot:
    currency: str
    title: str

    def __init__(self, currency: str):
        self.currency = currency
        self.title = self.__get_title(currency)

        plt.figure(figsize=(12, 7))
        ax = plt.axes()
        ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
        ax.xaxis.set_major_locator(DayLocator(interval=14))
        plt.xlabel('Data')
        plt.ylabel('Cena w USD')
        plt.title(self.title)
        plt.grid()

    def __get_title(self, currency: str) -> str:
        title = ('Wykres faktycznych cen {0}'
                 ' i przewidzianych wartości przez model').format(currency)
        return title

    def __get_prediction_label(self, model_data: ModelData) -> str:
        label = '{0} ('.format(model_data.name)
        for param in model_data.model_params:
            label += '{0}: {1}, '.format(param.name, param.value)
        label = label[:-2]
        label += ')'
        return label

    def add_data(self, data: PlotPriceData):
        plt.plot(data.dates, data.prices,
                 color=data.color, label=self.__get_prediction_label(data.model_data), linestyle=data.style)

    def show(self):
        plt.legend()
        plt.show()