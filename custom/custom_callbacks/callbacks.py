from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as backend
import tensorflow as tf
import numpy as np


class CustomCallback1(Callback):
    def __init__(self, dataset, settings, args):
        super().__init__()
        self.param1 = args["param1"]
        self.param2 = args["param2"]

    def on_epoch_end(self, epoch, logs=None):
        print("Processing epoch {}".format(epoch))

    def on_train_end(self, logs=None):
        pass

    def get_param1(self):
        return self.param1

    def get_param2(self):
        return self.param2


class CustomCallback2(Callback):
    def __init__(self, dataset, settings, args):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        print("Processing epoch {}".format(epoch))

    def on_train_end(self, logs=None):
        pass

    # def get_param1(self):
    #     return self.param1
    #
    # def get_param2(self):
    #     return self.param2


class SaveWeights(Callback):
    def __init__(self, dataset, settings, *args):
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append(self.model.get_weights())

    def get_weights(self):
        return self.weights
