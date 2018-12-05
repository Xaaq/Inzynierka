from matplotlib import pyplot as plt

from src.data_file_manager import DataFilesManager
from src.data_operator import DataOperator
from src.plotter import Plotter

if __name__ == "__main__":
    data_files_manager = DataFilesManager()
    plotter = Plotter()
    data_operator = DataOperator()

    input_data, output_data = data_files_manager.extract_simulation_means_data("simulation_output_data", slice(None, 1))
    plt.hist(output_data, bins=40)
    plt.ylabel("Ilość przypadków")
    plt.xlabel("Czas ewakuacji [s]")
    plt.show()
