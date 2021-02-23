from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import random


def mlp_grid_search(classes, number_of_samples, params, best_model=False):
    # set default values if not present in hyper_parameters_grid

    neurons = params["neurons"] if "neurons" in params else 50
    layers = params["layers"] if "layers" in params else 5
    activation = params["activation"] if "activation" in params else "relu"
    learning_rate = params["learning_rate"] if "learning_rate" in params else 0.001

    hp = {
        'neurons': neurons,
        'layers': layers,
        'activation': activation,
        'learning_rate': learning_rate
    }

    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(number_of_samples,)))
    for layer_index in range(layers - 1):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, hp


def cnn_grid_search(classes, number_of_samples, params, best_model=False):
    # set default values if not present in hyper_parameters_grid

    conv_layers = params["conv_layers"] if "conv_layers" in params else 1
    kernels = []
    strides = []
    filters = []
    pooling_type = []
    pooling_layers = []
    pooling_sizes = []
    pooling_strides = []
    for conv_layer in range(1, conv_layers + 1):
        kernels.append(params["kernel_{}".format(conv_layer)] if "kernel_{}".format(conv_layer) in params else 4)
        strides.append(params["stride_{}".format(conv_layer)] if "kernel_{}".format(conv_layer) in params else 1)
        filters.append(params["filters_{}".format(conv_layer)] if "filters_{}".format(conv_layer) in params else 16)
        pooling_sizes.append(params["pooling_size_{}".format(conv_layer)] if "pooling_size_{}".format(conv_layer) in params else 2)
        pooling_strides.append(params["pooling_stride_{}".format(conv_layer)] if "pooling_stride_{}".format(conv_layer) in params else 1)
        pooling_type.append(params["pooling_type_{}".format(conv_layer)] if "pooling_type_{}".format(conv_layer) in params else "Average")
        if pooling_type[conv_layer - 1] == "Average":
            pooling_layers.append(AveragePooling1D(pool_size=pooling_sizes[conv_layer - 1], strides=pooling_strides[conv_layer - 1]))
        elif pooling_type[conv_layer - 1] == "Max":
            pooling_layers.append(MaxPool1D(pool_size=pooling_sizes[conv_layer - 1], strides=pooling_strides[conv_layer - 1]))
        else:
            pooling_layers.append(AveragePooling1D(pool_size=pooling_sizes[conv_layer - 1], strides=pooling_strides[conv_layer - 1]))

    neurons = params["neurons"] if "neurons" in params else 50
    layers = params["layers"] if "layers" in params else 5
    activation = params["activation"] if "activation" in params else "relu"
    learning_rate = params["learning_rate"] if "learning_rate" in params else 0.001

    hp = {'conv_layers': conv_layers}
    for conv_layer in range(1, conv_layers + 1):
        hp["kernel_{}".format(conv_layer)] = kernels[conv_layer - 1]
        hp["stride_{}".format(conv_layer)] = strides[conv_layer - 1]
        hp["filters_{}".format(conv_layer)] = filters[conv_layer - 1]
        hp["pooling_size_{}".format(conv_layer)] = pooling_sizes[conv_layer - 1]
        hp["pooling_stride_{}".format(conv_layer)] = pooling_strides[conv_layer - 1]
        hp["pooling_type_{}".format(conv_layer)] = pooling_type[conv_layer - 1]
    hp["neurons"] = neurons
    hp["layers"] = layers
    hp["activation"] = activation
    hp["learning_rate"] = learning_rate

    model = Sequential()
    for conv_layer in range(1, conv_layers + 1):
        if conv_layer == 1:
            model.add(
                Conv1D(kernel_size=kernels[conv_layer - 1], strides=strides[conv_layer - 1], filters=filters[conv_layer - 1],
                       activation=activation, input_shape=(number_of_samples, 1)))
        else:
            model.add(
                Conv1D(kernel_size=kernels[conv_layer - 1], strides=strides[conv_layer - 1], filters=filters[conv_layer - 1],
                       activation=activation))
        model.add(pooling_layers[conv_layer - 1])
        model.add(BatchNormalization())
    model.add(Flatten())
    for layer_index in range(layers):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, hp
