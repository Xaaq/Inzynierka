import numpy as np
from matplotlib import pyplot as plt


class Plotter:
    @staticmethod
    def plot_history(history, save_path=None, max_y=400):
        plt.figure()
        plt.xlabel("Iteracja ucząca")
        plt.ylabel("Błąd średniokwadratowy")
        plt.plot(history.epoch, np.array(history.history["mean_squared_error"]), label="Dane treningowe")
        plt.plot(history.epoch, np.array(history.history["val_mean_squared_error"]), label="Dane testowe")
        plt.ylim([0, max_y])
        plt.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_learning_curves(history_list, save_path=None, max_y=400):
        data_count_list = [data_count for data_count, _ in history_list]
        mse_list = [history.history["mean_squared_error"][-1] for _, history in history_list]
        val_mse_list = [history.history["val_mean_squared_error"][-1] for _, history in history_list]

        plt.figure()
        plt.xlabel("Liczba zestawów danych uczących")
        plt.ylabel("Błąd średniokwadratowy")
        plt.plot(data_count_list, mse_list, label="Dane treningowe")
        plt.plot(data_count_list, val_mse_list, label="Dane testowe")
        plt.ylim([0, max_y])
        plt.legend()

        if save_path:
            plt.gcf().savefig(save_path, bbox_inches="tight")

        plt.show()
