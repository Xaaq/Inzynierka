import pickle

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


if __name__ == "__main__":
    data_files_manager = DataFilesManager()
    plotter = Plotter()
    data_operator = DataOperator()

    input_data, output_data = data_files_manager.extract_simulation_means_data("simulation_output_data", slice(None, 1),
                                                                               slice(None, None))
    filtered_data_indices = (output_data < 40).reshape((output_data.shape[0],))
    input_data, output_data = input_data[filtered_data_indices], output_data[output_data < 40]
    input_data, output_data = data_operator.permutate_data(input_data, output_data)
    input_data, data_mean, data_std = data_operator.normalize_data(input_data)

    # with open("data_mean_std.dat", "wb") as file_handler:
    #     pickle.dump((data_mean, data_std), file_handler)
    #
    # regularization_rate = 0.3
    # learning_epochs = 300
    # validation_split = 0.2
    #
    # # ===== Normal learning =====
    #
    # mse_model_list = []
    #
    # for _ in range(20):
    #     input_data, output_data = data_operator.permutate_data(input_data, output_data)
    #
    #     _, single_data_dimension = input_data.shape
    #     model = create_model(single_data_dimension, regularization_rate)
    #     history = model.fit(input_data, output_data, epochs=learning_epochs, validation_split=validation_split,
    #                         verbose=False)
    #
    #     output_mse = history.history["mean_squared_error"][-1]
    #     output_val_mse = history.history["val_mean_squared_error"][-1]
    #
    #     mse_difference = abs(output_mse - output_val_mse)
    #
    #     mse_model_list.append((mse_difference, model))
    #     print(output_mse, output_val_mse, mse_difference)
    #
    # mse_model_list.sort()
    # difference, model = mse_model_list[0]
    # model.save("saved_model_aaaa.hdf5")
    #
    # print(f"Saved model with {difference} difference")
