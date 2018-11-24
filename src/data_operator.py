import numpy as np


class DataOperator:
    @staticmethod
    def permutate_data(input_data: np.ndarray, output_data: np.ndarray):
        indices_permutation = np.random.permutation(input_data.shape[0])
        input_data = input_data[indices_permutation]
        output_data = output_data[indices_permutation]
        return input_data, output_data

    @staticmethod
    def normalize_data(data: np.ndarray):
        data_mean = data.mean(0)
        data_std = data.std(0)
        output_data = (data - data_mean) / data_std
        return output_data, data_mean, data_std
