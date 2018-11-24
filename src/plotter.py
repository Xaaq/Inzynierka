import numpy as np
from matplotlib import pyplot as plt


class Plotter:
    @staticmethod
    def plot_history(history):
        plt.figure()
        plt.xlabel("Iteracja ucząca")
        plt.ylabel("Błąd średniokwadratowy")
        plt.plot(history.epoch, np.array(history.history["mean_squared_error"]), label="Koszt danych treningowych")
        plt.plot(history.epoch, np.array(history.history["val_mean_squared_error"]), label="Koszt danych testowych")
        plt.ylim([0, 400])
        plt.legend()
        plt.show()

    @staticmethod
    def plot_learning_curves(history_list):
        data_count_list = [data_count for data_count, _ in history_list]
        mse_list = [history.history["mean_squared_error"][-1] for _, history in history_list]
        val_mse_list = [history.history["val_mean_squared_error"][-1] for _, history in history_list]

        plt.figure()
        plt.xlabel("Liczba zestawów danych uczących")
        plt.ylabel("Błąd średniokwadratowy")
        plt.plot(data_count_list, mse_list, label="Koszt danych treningowych")
        plt.plot(data_count_list, val_mse_list, label="Koszt danych testowych")
        plt.ylim([0, 400])
        plt.legend()
        plt.show()
