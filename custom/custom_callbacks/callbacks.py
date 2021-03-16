from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as backend
import tensorflow as tf
import numpy as np


class CustomCallback1(Callback):
    def __init__(self,
                 x_profiling, y_profiling, plaintexts_profiling,
                 ciphertext_profiling, key_profiling,
                 x_validation, y_validation, plaintexts_validation,
                 ciphertext_validation, key_validaton,
                 x_attack, y_attack, plaintexts_attack,
                 ciphertext_attack, key_attack,
                 param, leakage_model, key_rank_executions, key_rank_report_interval, key_rank_attack_traces,
                 *args):
        my_args = args[0]  # this line is mandatory
        self.param1 = my_args[0]
        self.param2 = my_args[1]

    def on_epoch_end(self, epoch, logs=None):
        print("Processing epoch {}".format(epoch))

    def on_train_end(self, logs=None):
        pass

    def get_param1(self):
        return self.param1

    def get_param2(self):
        return self.param2


class CustomCallback2(Callback):
    def __init__(self,
                 x_profiling, y_profiling, plaintexts_profiling,
                 ciphertext_profiling, key_profiling,
                 x_validation, y_validation, plaintexts_validation,
                 ciphertext_validation, key_validaton,
                 x_attack, y_attack, plaintexts_attack,
                 ciphertext_attack, key_attack,
                 param, leakage_model, key_rank_executions, key_rank_report_interval, key_rank_attack_traces,
                 *args):
        my_args = args[0]  # this line is mandatory
        # self.param1 = my_args[0]
        # self.param2 = my_args[1]

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
    def __init__(self,
                 x_profiling, y_profiling, plaintexts_profiling,
                 ciphertext_profiling, key_profiling,
                 x_validation, y_validation, plaintexts_validation,
                 ciphertext_validation, key_validaton,
                 x_attack, y_attack, plaintexts_attack,
                 ciphertext_attack, key_attack,
                 param, leakage_model, key_rank_executions, key_rank_report_interval, key_rank_attack_traces,
                 *args):
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append(self.model.get_weights())

    def get_weights(self):
        return self.weights


class PrunerCallback(Callback):
    def __init__(self,
                 x_profiling, y_profiling, plaintexts_profiling,
                 ciphertext_profiling, key_profiling,
                 x_validation, y_validation, plaintexts_validation,
                 ciphertext_validation, key_validaton,
                 x_attack, y_attack, plaintexts_attack,
                 ciphertext_attack, key_attack,
                 param, leakage_model, key_rank_executions, key_rank_report_interval, key_rank_attack_traces,
                 *args):
        my_args = args[0]  # this line is mandatory
        self.pruner = my_args[0]

    def on_train_end(self, logs=None):
        self.pruner.apply_pruning(self.model)

    def on_epoch_begin(self, epoch, logs=None):
        self.pruner.apply_pruning(self.model)


class ImplicitGradientRegularizationCallback(Callback):
    def __init__(self,
                 x_profiling, y_profiling, plaintexts_profiling,
                 ciphertext_profiling, key_profiling,
                 x_validation, y_validation, plaintexts_validation,
                 ciphertext_validation, key_validaton,
                 x_attack, y_attack, plaintexts_attack,
                 ciphertext_attack, key_attack,
                 param, leakage_model, key_rank_executions, key_rank_report_interval, key_rank_attack_traces,
                 *args):
        my_args = args[0]  # this line is mandatory
        self.current_epoch = 0
        self.x = x_profiling[0:1000]
        self.y = y_profiling[0:1000]
        self.param1 = my_args[0]
        self.number_of_epochs = my_args[0]
        self.implicit_gradient_regularization_epochs = np.zeros(self.number_of_epochs)

    def on_epoch_end(self, epoch, logs=None):
        input_trace = tf.Variable(self.x)

        with tf.GradientTape() as tape:
            tape.watch(input_trace)
            pred = self.model(input_trace)
            loss = tf.keras.losses.categorical_crossentropy(self.y, pred)

        grad = tape.gradient(loss, self.model.trainable_weights)

        implicit_gradient_regularization = 0

        for g_layer in grad:
            for index, g_connections in enumerate(g_layer):
                if isinstance(g_connections.numpy(), np.ndarray):
                    for g_neurons in g_connections.numpy():
                        implicit_gradient_regularization += g_neurons * g_neurons
                else:
                    implicit_gradient_regularization += g_connections.numpy() * g_connections.numpy()

        implicit_gradient_regularization /= (self.model.count_params() * len(self.x))
        self.implicit_gradient_regularization_epochs[epoch] = implicit_gradient_regularization

        backend.clear_session()

    def get_implicit_gradient_regularization_epochs(self):
        return self.implicit_gradient_regularization_epochs
