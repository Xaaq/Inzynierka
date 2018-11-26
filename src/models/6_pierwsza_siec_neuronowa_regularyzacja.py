import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data_file_manager import DataFilesManager
from src.data_operator import DataOperator
from src.plotter import Plotter


def create_model(input_neuron_count: int, regularization_rate: float, regularization_function):
    layers = [keras.layers.Dense(20, input_shape=(input_neuron_count,), activation=keras.activations.relu,
                                 kernel_regularizer=regularization_function(regularization_rate)),
              keras.layers.Dense(1, kernel_regularizer=regularization_function(regularization_rate))]

    model = keras.Sequential(layers)

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=tf.train.AdamOptimizer(0.1),
                  metrics=[keras.metrics.mean_squared_error])

    return model


if __name__ == "__main__":
    data_files_manager = DataFilesManager()
    plotter = Plotter()
    data_operator = DataOperator()

    for regularization_function in [keras.regularizers.l1, keras.regularizers.l2]:
        for regularization_rate in [0.1, 0.3, 1, 3]:
            input_data, output_data = data_files_manager.extract_simulation_means_data("simulation_output_data", 1)
            input_data, output_data = data_operator.permutate_data(input_data, output_data)
            input_data, _, _ = data_operator.normalize_data(input_data)

            learning_epochs = 300
            validation_split = 0.2

            # ===== Learning curves =====

            history_list = []
            number_of_splits = 20

            number_of_data_samples, _ = input_data.shape
            train_data_count = int(round((1 - validation_split) * number_of_data_samples))

            train_input_data = input_data[:train_data_count]
            train_output_data = output_data[:train_data_count]

            test_input_data = input_data[train_data_count:]
            test_output_data = output_data[train_data_count:]

            for i in range(number_of_splits)[1:]:
                i /= number_of_splits

                data_count, single_data_dimension = train_input_data.shape
                new_data_count = int(round(data_count * i))

                model = create_model(single_data_dimension, regularization_rate, regularization_function)
                history = model.fit(train_input_data[:new_data_count], train_output_data[:new_data_count],
                                    epochs=learning_epochs, validation_data=(test_input_data, test_output_data),
                                    verbose=False)
                history_list.append((new_data_count, history))

            plotter.plot_learning_curves(history_list, max_y=200)

            # ===== Normal learning =====

            mse_list = []
            history_mse_list = []

            for _ in range(10):
                input_data, output_data = data_operator.permutate_data(input_data, output_data)

                _, single_data_dimension = input_data.shape
                model = create_model(single_data_dimension, regularization_rate, regularization_function)
                history = model.fit(input_data, output_data, epochs=learning_epochs, validation_split=validation_split,
                                    verbose=False)

                output_mse = history.history["mean_squared_error"][-1]
                output_val_mse = history.history["val_mean_squared_error"][-1]

                mse_list.append([output_mse, output_val_mse])
                history_mse_list.append((output_val_mse, history))

            history_mse_list.sort()
            _, best_history = history_mse_list[0]
            plotter.plot_history(best_history, max_y=200)

            mean_mse = np.array(mse_list).mean(0)
            mse_std = np.array(mse_list).std(0)

            mean_str = f"{mean_mse[0]:.2f} ($\pm${mse_std[0]:.2f})".replace(".", ",")
            std_str = f"{mean_mse[1]:.2f} ($\pm${mse_std[1]:.2f})".replace(".", ",")
            print(regularization_function.__name__, regularization_rate, mean_str, std_str)
