import random
import numpy as np


# ---------------------------------------------------------------------------------------------------------------------#
#  Custom functions for data augmentation
# ---------------------------------------------------------------------------------------------------------------------#
def data_augmentation_shifts(data_set_samples, data_set_labels, batch_size, input_layer_shape):
    ns = len(data_set_samples[0])

    while True:

        x_train_shifted = np.zeros((batch_size, ns))
        rnd = random.randint(0, len(data_set_samples) - batch_size)
        x_mini_batch = data_set_samples[rnd:rnd + batch_size]

        for trace_index in range(batch_size):
            x_train_shifted[trace_index] = x_mini_batch[trace_index]
            shift = random.randint(-5, 5)
            if shift > 0:
                x_train_shifted[trace_index][0:ns - shift] = x_mini_batch[trace_index][shift:ns]
                x_train_shifted[trace_index][ns - shift:ns] = x_mini_batch[trace_index][0:shift]
            else:
                x_train_shifted[trace_index][0:abs(shift)] = x_mini_batch[trace_index][ns - abs(shift):ns]
                x_train_shifted[trace_index][abs(shift):ns] = x_mini_batch[trace_index][0:ns - abs(shift)]

        if len(input_layer_shape) == 3:
            x_train_shifted_reshaped = x_train_shifted.reshape((x_train_shifted.shape[0], x_train_shifted.shape[1], 1))
            yield x_train_shifted_reshaped, data_set_labels[rnd:rnd + batch_size]
        else:
            yield x_train_shifted, data_set_labels[rnd:rnd + batch_size]


def data_augmentation_gaussian_noise(data_set_samples, data_set_labels, batch_size, input_layer_shape):
    ns = len(data_set_samples[0])

    while True:

        x_train_augmented = np.zeros((batch_size, ns))
        rnd = random.randint(0, len(data_set_samples) - batch_size)
        x_mini_batch = data_set_samples[rnd:rnd + batch_size]

        noise = np.random.normal(0, 1, ns)

        for trace_index in range(batch_size):
            x_train_augmented[trace_index] = x_mini_batch[trace_index] + noise

        if len(input_layer_shape) == 3:
            x_train_augmented_reshaped = x_train_augmented.reshape((x_train_augmented.shape[0], x_train_augmented.shape[1], 1))
            yield x_train_augmented_reshaped, data_set_labels[rnd:rnd + batch_size]
        else:
            yield x_train_augmented, data_set_labels[rnd:rnd + batch_size]
