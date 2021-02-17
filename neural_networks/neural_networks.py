from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend


def cnn(classes, number_of_samples):
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=20, strides=1, activation='relu', padding='valid', input_shape=(number_of_samples, 1)))
    model.add(Conv1D(filters=8, kernel_size=10, strides=1, activation='relu', padding='valid', input_shape=(number_of_samples, 1)))
    # model.add(BatchNormalization())
    # model.add(AveragePooling1D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def mlp(classes, number_of_samples):
    model = Sequential()
    model.add(Dense(200, activation='selu', input_shape=(number_of_samples,)))
    model.add(Dense(200, activation='selu'))
    model.add(Dense(200, activation='selu'))
    model.add(Dense(200, activation='selu'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def mlp_large(classes, number_of_samples):
    model = Sequential()
    model.add(Dense(1000, activation='selu', input_shape=(number_of_samples,)))
    model.add(Dense(1000, activation='selu'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def mlp_es(classes, number_of_samples):
    model = Sequential()
    model.add(Dense(200, activation='selu', input_shape=(number_of_samples,)))
    model.add(Dense(200, activation='selu'))
    model.add(Dense(200, activation='selu'))
    model.add(Dense(200, activation='selu'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def mlp_mi_paper(classes, number_of_samples):
    model = Sequential()
    model.add(Dense(96, activation='relu', input_shape=(number_of_samples,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def cnn_es_chesctf(classes, number_of_samples):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=4, strides=4, activation='relu', padding='valid', input_shape=(number_of_samples, 1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dense(100, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def mlp_ascad_fixed_key(classes, number_of_samples):
    model = Sequential()
    model.add(Dense(100, activation='selu', input_shape=(number_of_samples,)))
    model.add(Dense(100, activation='selu'))
    model.add(Dense(100, activation='selu'))
    model.add(Dense(100, activation='selu'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
