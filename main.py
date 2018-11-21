import os
from copy import deepcopy

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from data_file_manager import DataFilesManager


def create_model(input_neuron_count: int):
    model = keras.Sequential([
        keras.layers.Dense(40, input_shape=(input_neuron_count,), activation=keras.activations.relu,
                           kernel_regularizer=keras.regularizers.l1_l2(3)),
        keras.layers.Dense(1)
    ])

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=tf.train.AdamOptimizer(0.1),
                  metrics=[keras.metrics.mean_squared_error])

    return model


def plot_history(history):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title("Learning over time")
    plt.plot(history.epoch, np.array(history.history["mean_squared_error"]), label="Train Loss")
    plt.plot(history.epoch, np.array(history.history["val_mean_squared_error"]), label="Validation train Loss")
    plt.ylim([0, 400])
    plt.legend()
    plt.show()


def plot_learning_curves(history_list):
    data_count_list = [data_count for data_count, _ in history_list]
    mse_list = [history.history["mean_squared_error"][-1] for _, history in history_list]
    val_mse_list = [history.history["val_mean_squared_error"][-1] for _, history in history_list]

    plt.figure()
    plt.xlabel("Number of data samples")
    plt.ylabel("Mean squared error")
    plt.title("Learning curves")
    plt.plot(data_count_list, mse_list, label="Train Loss")
    plt.plot(data_count_list, val_mse_list, label="Validation train Loss")
    plt.ylim([0, 400])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=4, threshold=np.nan, suppress=True)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    data_files_manager = DataFilesManager()
    input_data, output_data = data_files_manager.extract_simulation_means_data("simulation_output_data", 1)

    indices_permutation = np.random.permutation(input_data.shape[0])
    input_data = input_data[indices_permutation]
    output_data = output_data[indices_permutation]

    data_mean = input_data.mean(0)
    data_std = input_data.std(0)
    input_data = (input_data - data_mean) / data_std

    learning_epochs = 300
    validation_split = 0.2

    # ===== Learning curves =====

    history_list = []
    number_of_splits = 20

    for i in range(number_of_splits)[1:]:
        print(f"Iteration {i}/{number_of_splits - 1}")
        i /= number_of_splits

        data_count, single_data_dimension = input_data.shape
        new_data_count = int(round(data_count * i))

        model = create_model(single_data_dimension)
        history = model.fit(input_data[:new_data_count], output_data[:new_data_count], epochs=learning_epochs,
                            validation_split=validation_split, verbose=False)
        history_list.append((new_data_count, history))

    plot_learning_curves(history_list)

    # ===== Normal learning =====

    _, single_data_dimension = input_data.shape
    model = create_model(single_data_dimension)
    history = model.fit(input_data, output_data, epochs=learning_epochs, validation_split=validation_split,
                        verbose=False)
    plot_history(history)

    output_mse = history.history["mean_squared_error"][-1]
    output_val_mse = history.history["val_mean_squared_error"][-1]
    print(f"Output MSE: {output_mse}")
    print(f"Output validation MSE: {output_val_mse}")

    # ===== Lowest error finding =====

    input_output_list = []
    single_data_sample = np.array([input_data[0]])
    print(model.predict(single_data_sample))

    tf_gradient = tf.gradients(model.output, model.input)
    init = tf.global_variables_initializer()

    # single_data_sample_length = single_data_sample.shape[1]
    # parameter_index = 0

    for i in range(100):
        with tf.Session() as sess:
            sess.run(init)
            input_data_gradient = (sess.run(tf_gradient, feed_dict={model.input: single_data_sample}))[0] * 0.01
            single_data_sample -= input_data_gradient
            # single_data_sample[0][parameter_index] -= input_data_gradient[0][parameter_index]
            #
            # parameter_index += 1
            #
            # if parameter_index == single_data_sample_length:
            #     parameter_index = 0

        network_output = model.predict(single_data_sample)
        input_output_list.append((deepcopy(single_data_sample), network_output))
        print(network_output)

    input_output_list.sort(key=lambda value: value[1])

    for i, j in input_output_list[:10]:
        unnormalized_data = i * data_std + data_mean
        data_files_manager.create_parameters_file(unnormalized_data[0], f"generated_parameters/params_{j[0][0]}.xml")
        print(unnormalized_data, j)
