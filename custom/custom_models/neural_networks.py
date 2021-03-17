from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend


def cnn(classes, number_of_samples):
    model = Sequential(name="basic_cnn")
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
    model = Sequential(name="basic_mlp")
    model.add(Dense(200, activation='selu', input_shape=(number_of_samples,)))
    model.add(Dense(200, activation='selu'))
    model.add(Dense(200, activation='selu'))
    model.add(Dense(200, activation='selu'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def ASCAD_mlp(classes=256, number_of_samples=1400):
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(number_of_samples,)))
    for i in range(4):
        model.add(Dense(200, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    optimizer = RMSprop(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# CNN Best model
def ASCAD_cnn(classes=256, number_of_samples=1400):
    model = Sequential(name="ASCAD_cnn")
    # Block 1
    model.add(Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1', input_shape=(number_of_samples, 1)))
    model.add(AveragePooling1D(pool_size=2, strides=2, name='block1_pool'))
    # Block 2
    model.add(Conv1D(128, 11, strides=1, activation='relu', padding='same', name='block2_conv1'))
    model.add(AveragePooling1D(pool_size=2, strides=2, name='block2_pool'))
    # Block 3
    model.add(Conv1D(256, 11, strides=1, activation='relu', padding='same', name='block3_conv1'))
    model.add(AveragePooling1D(pool_size=2, strides=2, name='block3_pool'))
    # Block 4
    model.add(Conv1D(512, 11, strides=1, activation='relu', padding='same', name='block4_conv1'))
    model.add(AveragePooling1D(pool_size=2, strides=2, name='block4_pool'))
    # Block 5
    model.add(Conv1D(512, 11, strides=1, activation='relu', padding='same', name='block5_conv1'))
    model.add(AveragePooling1D(pool_size=2, strides=2, name='block5_pool'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dense(classes, activation='softmax', name='predictions'))
    model.summary()
    optimizer = RMSprop(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def methodology_cnn_ascad(classes=256, number_of_samples=700):
    model = Sequential(name="methodology_cnn_ascad")
    model.add(Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1',
                     input_shape=(number_of_samples, 1)))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=2, strides=2, name='block1_pool'))
    model.add(Flatten())
    model.add(Dense(10, activation='selu', kernel_initializer='he_uniform', name='fc1'))
    model.add(Dense(10, activation='selu', kernel_initializer='he_uniform', name='fc2'))
    model.add(Dense(classes, activation='softmax', name='predictions'))
    model.summary()
    optimizer = Adam(lr=5e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def methodology_cnn_aeshd(classes=256, number_of_samples=1250):
    model = Sequential(name="methodology_cnn_aeshd")
    model.add(Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1',
                     input_shape=(number_of_samples, 1)))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=2, strides=2, name='block1_pool'))
    model.add(Flatten())
    model.add(Dense(2, activation='selu', kernel_initializer='he_uniform', name='fc1'))
    model.add(Dense(classes, activation='softmax', name='predictions'))
    model.summary()
    optimizer = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def methodology_cnn_aesrd(classes=256, number_of_samples=700):
    model = Sequential(name="methodology_cnn_aesrd")

    # 1st convolutional block
    model.add(Conv1D(8, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1',
                     input_shape=(number_of_samples, 1)))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=2, strides=2, name='block1_pool'))

    # 2nd convolutional block
    model.add(Conv1D(16, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=50, strides=50, name='block2_pool'))

    # 3rd convolutional block
    model.add(Conv1D(32, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=7, strides=7, name='block3_pool'))

    model.add(Flatten())
    model.add(Dense(10, activation='selu', kernel_initializer='he_uniform', name='fc1'))
    model.add(Dense(10, activation='selu', kernel_initializer='he_uniform', name='fc2'))
    model.add(Dense(classes, activation='softmax', name='predictions'))
    model.summary()
    optimizer = Adam(lr=10e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def methodology_cnn_dpav4(classes=256, number_of_samples=4000):
    model = Sequential(name="methodology_cnn_dpav4")
    model.add(Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1',
                     input_shape=(number_of_samples, 1)))
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=2, strides=2, name='block1_pool'))
    model.add(Flatten())
    model.add(Dense(2, activation='selu', kernel_initializer='he_uniform', name='fc1'))
    model.add(Dense(classes, activation='softmax', name='predictions'))
    model.summary()
    optimizer = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
