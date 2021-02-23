from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as backend
import tensorflow as tf
import numpy as np
import importlib
from sklearn.metrics import confusion_matrix
from termcolor import colored
import os


class EarlyStoppingCallback(Callback):
    def __init__(self, x_profiling, y_profiling, plaintexts_profiling,
                 ciphertexts_profiling, key_profiling,
                 x_validation, y_validation, plaintexts_validation,
                 ciphertexts_validation, key_validation,
                 x_attack, y_attack, plaintexts_attack,
                 ciphertexts_attack, key_attack,
                 param, aes_leakage_model,
                 key_rank_executions, key_rank_report_interval, key_rank_attack_traces,
                 metrics=None, timestamp=None):
        # profiling
        self.x_profiling = x_profiling
        self.y_profiling = y_profiling
        self.plaintexts_profiling = plaintexts_profiling
        self.ciphertexts_profiling = ciphertexts_profiling
        self.key_profiling = key_profiling
        # validation
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.plaintexts_validation = plaintexts_validation
        self.ciphertexts_validation = ciphertexts_validation
        self.key_validation = key_validation
        # attack
        self.x_attack = x_attack
        self.y_attack = y_attack
        self.plaintexts_attack = plaintexts_attack
        self.ciphertexts_attack = ciphertexts_attack
        self.key_attack = key_attack

        self.param = param
        self.aes_leakage_model = aes_leakage_model
        self.key_rank_executions = key_rank_executions
        self.key_rank_report_interval = key_rank_report_interval
        self.key_rank_attack_traces = key_rank_attack_traces

        self.metrics = metrics
        self.timestamp = timestamp

        self.metric_results = {}
        for metric in self.metrics:
            self.metric_results[metric] = []

    def on_epoch_end(self, epoch, logs={}):

        if self.metrics is not None:

            for metric in self.metrics:
                script = importlib.import_module("custom.custom_metrics.{}".format(self.metrics[metric]["class"]))
                metric_value = script.run(self.x_profiling, self.y_profiling, self.plaintexts_profiling,
                                          self.ciphertexts_profiling, self.key_profiling,
                                          self.x_validation, self.y_validation, self.plaintexts_validation,
                                          self.ciphertexts_validation, self.key_validation,
                                          self.x_attack, self.y_attack, self.plaintexts_attack,
                                          self.ciphertexts_attack, self.key_attack,
                                          self.param, self.aes_leakage_model,
                                          self.key_rank_executions, self.key_rank_report_interval, self.key_rank_attack_traces,
                                          self.model, self.metrics[metric]["parameters"])
                self.metric_results[metric].append(metric_value)

                if epoch > 0:
                    ref_value_epoch_list = None
                    ref_metric = None
                    if self.metrics[metric]["direction"] == "max":
                        if isinstance(metric_value, list):
                            ref_value_epoch_list = np.zeros(len(self.metric_results[metric][epoch]))
                            for index, current_epoch_value in enumerate(self.metric_results[metric][epoch]):
                                ref_value_epoch_list[index] = np.max(self.metric_results[metric][0:epoch])
                        else:
                            ref_metric = np.max(self.metric_results[metric][0:epoch])
                    else:
                        if isinstance(metric_value, list):
                            ref_value_epoch_list = np.zeros(len(self.metric_results[metric][epoch]))
                            for index, current_epoch_value in enumerate(self.metric_results[metric][epoch]):
                                ref_value_epoch_list[index] = np.min(self.metric_results[metric][0:epoch])
                        else:
                            ref_metric = np.min(self.metric_results[metric][0:epoch])

                    if isinstance(metric_value, list):
                        for index, current_epoch_value in enumerate(self.metric_results[metric][epoch]):
                            if (self.metric_results[metric][epoch][index] > ref_value_epoch_list[index] and self.metrics[metric][
                                "direction"] == "max") or (
                                    self.metric_results[metric][epoch][index] < ref_value_epoch_list[index] and self.metrics[metric][
                                "direction"] == "min"):
                                os.remove(
                                    "../resources/models/best_model_{}_{}_{}.h5".format(metric, self.timestamp, index))
                                self.model.save_weights(
                                    "../resources/models/best_model_{}_{}_{}.h5".format(metric, self.timestamp, index))
                                print(colored(
                                    "\nmodel saved {} = {} at epoch {}\n".format(metric, self.metric_results[metric][epoch][index], epoch),
                                    "blue"))
                    else:
                        if (self.metric_results[metric][epoch] > ref_metric and self.metrics[metric]["direction"] == "max") or (
                                self.metric_results[metric][epoch] < ref_metric and self.metrics[metric]["direction"] == "min"):
                            os.remove("../resources/models/best_model_{}_{}.h5".format(metric, self.timestamp))
                            self.model.save_weights("../resources/models/best_model_{}_{}.h5".format(metric, self.timestamp))
                            print(colored("\nmodel saved {} = {} at epoch {}\n".format(metric, self.metric_results[metric][epoch], epoch),
                                          "blue"))
                else:
                    if isinstance(metric_value, list):
                        for index, current_epoch_value in enumerate(self.metric_results[metric][epoch]):
                            self.model.save_weights(
                                "../resources/models/best_model_{}_{}_{}.h5".format(metric, self.timestamp, index))
                            print(colored(
                                "\nmodel saved {} = {} at epoch {}\n".format(metric, self.metric_results[metric][epoch][index], epoch),
                                "blue"))
                    else:
                        self.model.save_weights("../resources/models/best_model_{}_{}.h5".format(metric, self.timestamp))
                        print(colored("\nmodel saved {} = {} at epoch {}\n".format(metric, self.metric_results[metric][epoch], epoch),
                                      "blue"))

    def get_metric_results(self):
        return self.metric_results


class InputGradients(Callback):
    def __init__(self, x_data, y_data, number_of_epochs):
        self.current_epoch = 0
        self.x = x_data
        self.y = y_data
        self.number_of_samples = len(x_data[0])
        self.number_of_epochs = number_of_epochs
        self.gradients = np.zeros((number_of_epochs, self.number_of_samples))
        self.gradients_sum = np.zeros(self.number_of_samples)

    def on_epoch_end(self, epoch, logs=None):
        input_trace = tf.Variable(self.x)

        with tf.GradientTape() as tape:
            tape.watch(input_trace)
            pred = self.model(input_trace)
            loss = tf.keras.losses.categorical_crossentropy(self.y, pred)

        grad = tape.gradient(loss, input_trace)

        input_gradients = np.zeros(self.number_of_samples)
        for i in range(len(self.x)):
            input_gradients += grad[i].numpy().reshape(self.number_of_samples)

        self.gradients[epoch] = input_gradients
        if np.max(self.gradients[epoch]) != 0:
            self.gradients_sum += np.abs(self.gradients[epoch] / np.max(self.gradients[epoch]))
        else:
            self.gradients_sum += np.abs(self.gradients[epoch])

        backend.clear_session()

    def grads(self):
        return np.abs(self.gradients_sum)

    def grads_epoch(self):
        for e in range(self.number_of_epochs):
            if np.max(self.gradients[e]) != 0:
                self.gradients[e] = np.abs(self.gradients[e] / np.max(self.gradients[e]))
            else:
                self.gradients[e] = np.abs(self.gradients[e])
        return self.gradients


class ConfusionMatrix(Callback):
    def __init__(self, x, y_true):
        self.current_epoch = 0
        self.x = x
        self.y_true = np.argmax(y_true, axis=1)
        self.cm = None

    def on_epoch_end(self, epoch, logs=None):
        Y_pred = self.model.predict(self.x)
        y_pred = np.argmax(Y_pred, axis=1)
        self.cm = confusion_matrix(y_true=self.y_true, y_pred=y_pred)

    def get_confusion_matrix(self):
        return self.cm
