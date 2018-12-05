import pickle
from copy import deepcopy
from random import randint

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data_file_manager import DataFilesManager
from src.data_operator import DataOperator
from src.plotter import Plotter


def create_model(input_neuron_count: int, regularization_rate: float):
    layers = [keras.layers.Dense(20, input_shape=(input_neuron_count,), activation=keras.activations.relu,
                                 kernel_regularizer=keras.regularizers.l1(regularization_rate)),
              keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l1(regularization_rate))]

    model = keras.Sequential(layers)

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(0.1),
                  metrics=[keras.metrics.mean_squared_error])

    return model


def compute_gradient(single_data_sample, index):
    epsilon = 0.1
    plus_epsilon = deepcopy(single_data_sample)
    plus_epsilon[0][index] += epsilon

    minus_epsilon = deepcopy(single_data_sample)
    minus_epsilon[0][index] -= epsilon

    plus_epsilon = model.predict(plus_epsilon)
    minus_epsilon = model.predict(minus_epsilon)
    gradient = (plus_epsilon - minus_epsilon) / (2 * epsilon)
    return gradient


if __name__ == "__main__":
    data_files_manager = DataFilesManager()
    plotter = Plotter()
    data_operator = DataOperator()

    with open("data_mean_std.dat", "rb") as file_handler:
        data_mean, data_std = pickle.load(file_handler)

    random_index = randint(0, 399)
    input_data, output_data = data_files_manager.extract_simulation_means_data("simulation_output_data",  slice(None, 1),
                                                                               slice(random_index, random_index + 1))
    input_data = (input_data - data_mean) / data_std

    model = keras.models.load_model("saved_model_evacuation_time.hdf5")

    print(input_data, output_data)

    single_data_sample = np.array([input_data[0]])
    print(f"Predicted {model.predict(single_data_sample)}")

    for _ in range(100):
        for i, (n, (min, max), _) in enumerate(DataFilesManager.independent_parameters):
            gradient = compute_gradient(single_data_sample, i)

            after_gradient = single_data_sample[0][i] - gradient
            single_data_sample[0][i] = ((np.clip(after_gradient * data_std[i] + data_mean[i], min, max) - data_mean[i])
                                        / data_std[i])

        for i, (n1, n2, (min, max), _) in enumerate(DataFilesManager.dependent_parameters):
            i = (i * 2) + len(DataFilesManager.independent_parameters)

            gradient = compute_gradient(single_data_sample, i)

            after_gradient = single_data_sample[0][i] - gradient
            single_data_sample[0][i] = ((np.clip(after_gradient * data_std[i] + data_mean[i], min, max) - data_mean[i])
                                        / data_std[i])

            gradient = compute_gradient(single_data_sample, i + 1)

            after_gradient = single_data_sample[0][i + 1] - gradient
            single_data_sample[0][i + 1] = (np.clip(after_gradient * data_std[i + 1] + data_mean[i + 1], 0,
                                                    (single_data_sample[0][i] * data_std[i] + data_mean[i]) / 5) -
                                            data_mean[i + 1]) / data_std[i + 1]

    output_parameters = single_data_sample * data_std + data_mean
    data_files_manager.create_parameters_file(output_parameters[0], "tested_output_data/data.xml")

    print(output_parameters)
    print(model.predict(single_data_sample))
