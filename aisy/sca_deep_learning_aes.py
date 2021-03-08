import pandas as pd
import time
import itertools
import json
from tensorflow.keras.utils import to_categorical
from custom.custom_datasets.datasets import *
from aisy.sca_callbacks import *
from aisy.sca_keras_models import ScaKerasModels
from aisy.sca_functions import ScaFunctions
from aisy.sca_database_inserts import ScaDatabaseInserts
from aisy.sca_load_datasets import ScaLoadDatasets
from aisy.sca_aes_create_intermediates import *
from app import databases_root_folder, datasets_root_folder
from aisy.neural_networks_grid_search import *
from aisy.neural_networks_random_search import *
from termcolor import colored
from sklearn.utils import shuffle


class AisyAes:

    def __init__(self):

        self.settings = None
        self.datasets_root_folder = datasets_root_folder
        self.database_root_folder = databases_root_folder
        self.target_params = {}
        self.leakage_model = None
        self.callbacks = []
        self.custom_callbacks = {}
        self.model = None
        self.model_obj = None
        self.model_name = None
        self.model_class = None
        self.database_name = None
        self.db_inserts = None
        self.epochs = 50
        self.batch_size = 400
        self.classes = 9

        self.ge_attack = None
        self.sr_attack = None
        self.ge_validation = None
        self.sr_validation = None
        self.metric_profiling = []
        self.metric_validation = []
        self.metric_attack = []

        self.ge_all_validation = None
        self.sr_all_validation = None
        self.ge_all_attack = None
        self.sr_all_attack = None
        self.output_probabilities_all_models = None
        self.output_probabilities = None
        self.ge_best_model_validation = None
        self.ge_best_model_attack = None
        self.sr_best_model_validation = None
        self.sr_best_model_attack = None
        self.ge_ensemble = None
        self.ge_ensemble_best_models = None
        self.sr_ensemble = None
        self.sr_ensemble_best_models = None
        self.ge_attack_early_stopping = None
        self.sr_attack_early_stopping = None

        self.hyper_parameters = []
        self.hyper_parameters_search = []
        self.learning_rate = None
        self.optimizer = None

        self.key_rank_executions = 1
        self.key_rank_report_interval = 10
        self.key_rank_attack_traces = 1000
        self.sr_runs = 1
        self.key_int = None

        self.visualization_active = False
        self.data_augmentation_active = False
        self.ensemble_active = False
        self.grid_search_active = False
        self.random_search_active = False
        self.early_stopping_active = False
        self.early_stopping_metrics = None
        self.confusion_matrix_active = False
        self.compute_ge_active = True
        self.timestamp = 0
        self.save_database = True
        self.save_to_npz = False
        self.probability_rank_plot = False

        self.z_score_mean = None
        self.z_score_std = None
        self.z_norm = False

        self.callback_key_rank_validation = None
        self.callback_input_gradients = None
        self.callback_early_stopping = None
        self.callback_confusion_matrix = None

        self.start = None

        self.leakage_model_is_set = False
        self.database_is_set = False
        self.dataset_is_set = False

    def set_datasets_root_folder(self, datasets_root_folder):
        self.datasets_root_folder = datasets_root_folder

    def set_dataset(self, target):
        self.target_params = datasets_dict[target]

    def set_dataset_filename(self, dataset_filename):
        self.target_params["filename"] = dataset_filename

    def set_database_root_folder(self, database_root_folder):
        if "/" in database_root_folder and database_root_folder[len(database_root_folder) - 1] != "/":
            database_root_folder += "/"
        if "\\" in database_root_folder and database_root_folder[len(database_root_folder) - 1] != "\\":
            database_root_folder += "\\"
        print(database_root_folder)
        self.database_root_folder = database_root_folder

    def set_database_name(self, database_name):
        self.database_name = database_name

    def set_znorm(self):
        self.z_norm = True

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_key(self, key):
        self.target_params["key"] = key

    def set_number_of_profiling_traces(self, profiling_traces):
        self.target_params["number_of_profiling_traces"] = profiling_traces

    def set_number_of_attack_traces(self, attack_traces):
        self.target_params["number_of_attack_traces"] = attack_traces

    def set_first_sample(self, first_sample):
        self.target_params["first_sample"] = first_sample

    def set_number_of_samples(self, number_of_samples):
        self.target_params["number_of_samples"] = number_of_samples

    def set_aes_leakage_model(self, leakage_model="HW", bit=0, byte=0, round=1, round_first=1, round_second=1, cipher="AES128",
                              target_state="Sbox", target_state_first="Sbox", target_state_second="Sbox",
                              direction="Encryption", attack_direction="input"):

        """
        Function to set the AES Leakage Model in the profiled SCA execution.
        :parameter
            leakage_model: 'HW', 'ID' or 'bit'
            bit: index of target bit (min 0, max 7)
            byte: index of target key byte
            round: index the target round
            round_first: index the first target round
            round_second: index the second target seround
            target_state: 'Sbox', InvSbox', 'AddRoundKey', 'MixColumns', 'InvMixColumns', 'ShiftRows', 'InvShiftRows'
            target_state_first: 'Input', 'Sbox', InvSbox', 'AddRoundKey', 'MixColumns', 'InvMixColumns', 'ShiftRows', 'InvShiftRows',
            'Output'
            target_state_second: 'Input', 'Sbox', InvSbox', 'AddRoundKey', 'MixColumns', 'InvMixColumns', 'ShiftRows', 'InvShiftRows',
            'Output'
            attack_direction: input, output
            direction: 'Encryption', 'Decryption'

        :return
            dictionary containing AES leakage model information:

            self.leakage_model = {
                "leakage_model": leakage_model,
                "bit": bit,
                "byte": byte,
                "round": round,
                "round_first": round_first,
                "round_second": round_second,
                "target_state": target_state,
                "target_state_first": target_state_first,
                "target_state_second": target_state_second,
                "direction": direction,
                "attack_direction": input
            }

        """

        self.leakage_model = {
            "leakage_model": leakage_model,
            "bit": bit,
            "byte": byte,
            "round": round,
            "round_first": round_first,  # for Hamming Distance
            "round_second": round_second,  # for Hamming Distance
            "cipher": cipher,
            "target_state": target_state,
            "target_state_first": target_state_first,  # for Hamming Distance
            "target_state_second": target_state_second,  # for Hamming Distance
            "direction": direction,
            "attack_direction": attack_direction
        }

        if self.target_params is not None:
            if self.leakage_model["leakage_model"] == "HW" or self.leakage_model["leakage_model"] == "HD":
                self.classes = 9
            elif self.leakage_model["leakage_model"] == "ID":
                self.classes = 256
            else:
                self.classes = 2
        else:
            print("Parameters (param) from target is not selected. Set target before the leakage model.")

        self.leakage_model_is_set = True

        return self.leakage_model

    def add_callback(self, callback):
        self.callbacks.append(callback)
        return self.callbacks

    def add_custom_callback(self, callback, callback_name):
        self.custom_callbacks[callback_name] = callback

    def get_custom_callbacks(self):
        return self.custom_callbacks

    def create_z_score_norm(self, dataset):
        self.z_score_mean = np.mean(dataset, axis=0)
        self.z_score_std = np.std(dataset, axis=0)

    def apply_z_score_norm(self, dataset):
        for index in range(len(dataset)):
            dataset[index] = (dataset[index] - self.z_score_mean) / self.z_score_std

    def set_neural_network(self, model):

        import inspect
        if inspect.isfunction(model):
            self.model_class = model
            if "number_of_samples" not in self.target_params.keys():
                print("ERROR: Dataset 'number_of_samples' not specified. Please use set_number_of_samples method to specify it.")
                return

            self.model_name = model.__name__
            self.model = model(self.classes, self.target_params["number_of_samples"])
        else:
            self.model_class = type(model)
            self.model_name = model.name
            self.model = model

    def get_model(self):
        return self.model

    def set_model_weights(self, weights):
        self.model.set_weights(weights)

    def get_db_inserts(self):
        return self.db_inserts

    def get_analysis_id(self):
        return self.db_inserts.get_analysis_id()

    def get_metrics_profiling(self):
        return self.metric_profiling

    def get_metrics_validation(self):
        return self.metric_validation

    def get_metrics_attack(self):
        return self.metric_attack

    def get_test_guessing_entropy(self):
        return self.ge_attack

    def get_test_success_rate(self):
        return self.sr_attack

    def initialize_result_vectors(self, nt_kr):
        self.ge_attack = np.zeros(nt_kr)
        self.sr_attack = np.zeros(nt_kr)

    def train_model(self, x_profiling, y_profiling, x_attack, y_attack,
                    plaintext_profiling, plaintext_attack,
                    ciphertext_profiling, ciphertext_attack,
                    key_profiling, key_attack,
                    data_augmentation, visualization, key_rank_report_interval, key_rank_attack_traces, search_index=None,
                    x_validation=None,
                    y_validation=None,
                    plaintext_validation=None,
                    ciphertext_validation=None,
                    key_validation=None,
                    custom_callbacks=None,
                    train_best_model=False):

        # reshape if needed
        input_layer_shape = self.model.get_layer(index=0).input_shape
        if len(input_layer_shape) == 3:
            x_profiling_reshaped = x_profiling.reshape((x_profiling.shape[0], x_profiling.shape[1], 1))
            if self.early_stopping_active or self.ensemble_active or self.grid_search_active or self.random_search_active:
                x_validation_reshaped = x_validation.reshape((x_validation.shape[0], x_validation.shape[1], 1))
            else:
                x_validation_reshaped = None
            x_attack_reshaped = x_attack.reshape((x_attack.shape[0], x_attack.shape[1], 1))
        else:
            x_profiling_reshaped = x_profiling
            if self.early_stopping_active or self.ensemble_active or self.grid_search_active or self.random_search_active:
                x_validation_reshaped = x_validation
            else:
                x_validation_reshaped = None
            x_attack_reshaped = x_attack

        # callbacks
        self.callbacks = []
        if self.visualization_active:
            self.callback_input_gradients = InputGradients(x_profiling_reshaped[0:visualization[0]], y_profiling[0:visualization[0]],
                                                           self.epochs)
            self.add_callback(self.callback_input_gradients)
        if self.early_stopping_active:
            self.timestamp = str(time.time()).replace(".", "")
            self.callback_early_stopping = EarlyStoppingCallback(x_profiling_reshaped, y_profiling, plaintext_profiling,
                                                                 ciphertext_profiling, key_profiling,
                                                                 x_validation_reshaped, y_validation, plaintext_validation,
                                                                 ciphertext_validation, key_validation,
                                                                 x_attack_reshaped, y_attack, plaintext_attack,
                                                                 ciphertext_attack, key_attack,
                                                                 self.target_params, self.leakage_model, self.key_rank_executions,
                                                                 key_rank_report_interval, key_rank_attack_traces,
                                                                 self.early_stopping_metrics, self.timestamp)
            self.add_callback(self.callback_early_stopping)
        if self.confusion_matrix_active:
            self.callback_confusion_matrix = ConfusionMatrix(x_attack, y_attack)
            self.add_callback(self.callback_confusion_matrix)
        if custom_callbacks is not None:
            for custom_callback in custom_callbacks:
                module_name = importlib.import_module("custom.custom_callbacks.callbacks")
                custom_callback_class = getattr(module_name, custom_callback['class'])
                # custom_callback_class = importlib.import_module("custom.custom_callbacks.callbacks.{}".format(custom_callback['class']))
                custom_callback_obj = custom_callback_class(x_profiling_reshaped, y_profiling, plaintext_profiling,
                                                            ciphertext_profiling, key_profiling,
                                                            x_validation_reshaped, y_validation, plaintext_validation,
                                                            ciphertext_validation, key_validation,
                                                            x_attack_reshaped, y_attack, plaintext_attack,
                                                            ciphertext_attack, key_attack,
                                                            self.target_params, self.leakage_model, self.key_rank_executions,
                                                            key_rank_report_interval, key_rank_attack_traces,
                                                            custom_callback['parameters'])
                self.add_callback(custom_callback_obj)
                self.add_custom_callback(custom_callback_obj, custom_callback['class'])
        callbacks = self.callbacks

        if self.data_augmentation_active:
            da_method = data_augmentation[0](x_profiling, y_profiling, self.batch_size, input_layer_shape)
            history = self.model.fit_generator(
                generator=da_method,
                steps_per_epoch=data_augmentation[1],
                epochs=self.epochs,
                verbose=2,
                validation_data=(x_attack_reshaped, y_attack),
                validation_steps=1,
                callbacks=callbacks)
        else:
            history = self.model.fit(
                x=x_profiling_reshaped,
                y=y_profiling,
                batch_size=self.batch_size,
                verbose=2,
                epochs=self.epochs,
                shuffle=True,
                validation_data=(x_attack_reshaped, y_attack),
                callbacks=callbacks)

        if self.compute_ge_active:
            self.compute_ge_and_sr(x_attack_reshaped,
                                   plaintext_attack, ciphertext_attack,
                                   key_rank_report_interval, key_rank_attack_traces,
                                   x_validation=x_validation_reshaped,
                                   plaintext_validation=plaintext_validation,
                                   ciphertext_validation=ciphertext_validation,
                                   early_stopping_metric_results=self.callback_early_stopping.get_metric_results() if self.early_stopping_active else None,
                                   best_model=train_best_model)

        if self.early_stopping_active:
            for early_stopping_metric in self.early_stopping_metrics:
                if isinstance(self.callback_early_stopping.get_metric_results()[early_stopping_metric][0], list):
                    for i in range(len(self.callback_early_stopping.get_metric_results()[early_stopping_metric][0])):
                        os.remove(
                            "../resources/models/best_model_{}_{}_{}.h5".format(early_stopping_metric, self.timestamp, i))
                else:
                    os.remove("../resources/models/best_model_{}_{}.h5".format(early_stopping_metric, self.timestamp))
            self.get_metrics_results(history, search_index, metric_results=self.callback_early_stopping.get_metric_results())
        else:
            self.get_metrics_results(history, search_index)

        return history

    def run_search(self, x_profiling, y_profiling, x_attack, y_attack,
                   plaintext_profiling, plaintext_attack,
                   ciphertext_profiling, ciphertext_attack,
                   key_profiling, key_attack,
                   data_augmentation, visualization,
                   key_rank_report_interval, key_rank_attack_traces, grid_search, random_search, x_validation=None, y_validation=None,
                   plaintext_validation=None, ciphertext_validation=None, key_validation=None, custom_callbacks=None):

        search_hp_combinations = None
        if self.grid_search_active:
            search_type = "Grid Search"
            hp_values = grid_search["hyper_parameters_search"]
            keys, values = zip(*hp_values.items())
            search_hp_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            max_trials = len(search_hp_combinations)
            search_metric = grid_search["metric"]
            stop_condition = grid_search["stop_condition"]
            stop_value = grid_search["stop_value"]
            train_after_search = grid_search["train_after_search"]
        else:
            search_type = "Random Search"
            hp_values = random_search["hyper_parameters_search"]
            max_trials = random_search["max_trials"]
            search_metric = random_search["metric"]
            stop_condition = random_search["stop_condition"]
            stop_value = random_search["stop_value"]
            train_after_search = random_search["train_after_search"]

        print(colored("Total number of trials in hyperparameter search: {}".format(max_trials), "magenta"))

        ge_search = []
        sr_search = []
        loss_search = []
        acc_search = []

        ge_search_early_stopping = []
        sr_search_early_stopping = []

        hp_searches = []
        best_model_index = None
        hp_ids = []
        nt_key_rank = int(key_rank_attack_traces / key_rank_report_interval)

        search_model = None

        for search_index in range(max_trials):
            if self.grid_search_active:
                search_model = cnn_grid_search if grid_search["neural_network"] == "cnn" else mlp_grid_search
                self.model, hp = search_model(self.classes, self.target_params["number_of_samples"],
                                              search_hp_combinations[search_index])
            else:
                search_model = cnn_random_search if random_search["neural_network"] == "cnn" else mlp_random_search
                self.model, hp = search_model(self.classes, self.target_params["number_of_samples"], hp_values)

            if "epochs" in hp_values:
                if self.grid_search_active:
                    self.epochs = random.choice(hp_values["epochs"])
                else:
                    self.epochs = random.randrange(hp_values["epochs"]["min"], hp_values["epochs"]["max"] + hp_values["epochs"]["step"],
                                                   hp_values["epochs"]["step"])
                hp["epochs"] = self.epochs
            if "mini_batch" in hp_values:
                if self.grid_search_active:
                    self.batch_size = random.choice(hp_values["mini_batch"])
                else:
                    self.batch_size = random.randrange(hp_values["mini_batch"]["min"], hp_values["mini_batch"]["max"],
                                                       hp_values["mini_batch"]["step"])
                hp["mini_batch"] = self.batch_size

            hp_searches.append(hp)

            print(colored("Hyper-Parameters for Search {}: {}".format(search_index, json.dumps(hp, sort_keys=True, indent=4)), "blue"))

            history = self.train_model(x_profiling, y_profiling, x_attack, y_attack,
                                       plaintext_profiling, plaintext_attack,
                                       ciphertext_profiling, ciphertext_attack,
                                       key_profiling, key_attack,
                                       data_augmentation, visualization,
                                       key_rank_report_interval, key_rank_attack_traces, search_index=str(search_index),
                                       x_validation=x_validation, y_validation=y_validation,
                                       plaintext_validation=plaintext_validation,
                                       ciphertext_validation=ciphertext_validation,
                                       key_validation=key_validation,
                                       custom_callbacks=custom_callbacks)
            ge_search.append(self.ge_validation[nt_key_rank - 1])
            sr_search.append(self.sr_validation[nt_key_rank - 1])
            loss_search.append(history.history["loss"])
            acc_search.append(history.history["accuracy"])

            if self.probability_rank_plot:
                rank_results, output_probabilities = ScaFunctions().get_probability_ranks(x_attack, plaintext_attack, ciphertext_attack,
                                                                                          self.key_rank_attack_traces,
                                                                                          self.classes, self.leakage_model,
                                                                                          self.target_params, self.model)
                self.output_probabilities.append(output_probabilities)
                if self.save_database:
                    self.__save_probability_ranks(rank_results,
                                                  "Attack Byte {} Model Search {}".format(self.leakage_model["byte"], search_index))
                self.save_probability_ranks_figure(rank_results,
                                                   "Attack Byte {} Model Search {}".format(self.leakage_model["byte"], search_index))

            if self.early_stopping_active:
                ge_search_early_stopping.append(self.ge_attack_early_stopping)
                sr_search_early_stopping.append(self.sr_attack_early_stopping)

            model_name = "mlp_grid_search" if self.grid_search_active else "mlp_random_search"
            self.learning_rate = backend.eval(self.model.optimizer.lr)
            self.optimizer = self.model.optimizer.__class__.__name__

            self.set_hyper_parameters_search(hp_searches[search_index], self.ge_validation[nt_key_rank - 1])

            if search_index == 0:
                hp_ids.append(self.save_results_in_database(time.time() - self.start, model_name, hyperparameters_search=True))
            else:
                hp_ids.append(self.save_results_in_database(time.time() - self.start, model_name, update=True, hyperparameters_search=True))
            if self.save_database:
                self.save_results(search_index=search_index)
            if self.early_stopping_active and self.save_database:
                self.save_early_stopping_results(search_index=search_index)
            self.hyper_parameters_search = []
            backend.clear_session()

            if stop_condition:

                if search_metric == "guessing_entropy" and self.ge_validation[nt_key_rank - 1] <= stop_value:
                    best_model_index = search_index
                    print(colored("\nBest Model: {}".format(json.dumps(hp, sort_keys=True, indent=4)), "green"))
                    break
                if search_metric == "loss" and history.history["loss"] <= stop_value:
                    best_model_index = search_index
                    print(colored("\nBest Model: {}".format(json.dumps(hp, sort_keys=True, indent=4)), "green"))
                    break
                if search_metric == "accuracy" and history.history["accuracy"] >= stop_value:
                    best_model_index = search_index
                    print(colored("\nBest Model: {}".format(json.dumps(hp, sort_keys=True, indent=4)), "green"))
                    break

        if len(ge_search) == max_trials and best_model_index is None:

            if search_metric == "guessing_entropy":
                best_model_index = ge_search.index(min(ge_search))
                print(colored("\nBest Model: {}".format(json.dumps(hp_searches[best_model_index], sort_keys=True, indent=4)), "green"))
            if search_metric == "loss":
                best_model_index = loss_search.index(min(loss_search))
                print(colored("\nBest Model: {}".format(json.dumps(hp_searches[best_model_index], sort_keys=True, indent=4)), "green"))
            if search_metric == "accuracy":
                best_model_index = acc_search.index(max(acc_search))
                print(colored("\nBest Model: {}".format(json.dumps(hp_searches[best_model_index], sort_keys=True, indent=4)), "green"))

        if train_after_search:
            self.model, _ = search_model(self.classes, self.target_params["number_of_samples"], hp_searches[best_model_index],
                                         best_model=True)
            if "epochs" in hp_searches[best_model_index]:
                self.epochs = hp_searches[best_model_index]["epochs"]
            if "mini_batch" in hp_searches[best_model_index]:
                self.batch_size = hp_searches[best_model_index]["mini_batch"]

            self.train_model(x_profiling, y_profiling, x_attack, y_attack,
                             plaintext_profiling, plaintext_attack,
                             ciphertext_profiling, ciphertext_attack,
                             key_profiling, key_attack,
                             data_augmentation, visualization,
                             key_rank_report_interval, key_rank_attack_traces, search_index="best",
                             x_validation=x_validation,
                             y_validation=y_validation,
                             plaintext_validation=plaintext_validation,
                             ciphertext_validation=ciphertext_validation,
                             key_validation=key_validation,
                             custom_callbacks=custom_callbacks,
                             train_best_model=True)

            model_name = "mlp_grid_search" if self.grid_search_active else "mlp_random_search"
            self.learning_rate = backend.eval(self.model.optimizer.lr)
            self.optimizer = self.model.optimizer.__class__.__name__
            self.set_hyper_parameters_search(hp_searches[best_model_index], self.ge_attack[nt_key_rank - 1])
            hp_id = self.save_results_in_database(time.time() - self.start, model_name, update=True, hyperparameters_search=True)
            if self.save_database:
                self.save_results(best_model_search=True)
            if self.early_stopping_active and self.save_database:
                self.save_early_stopping_results(best_model_search=True)
        else:
            hp_id = hp_ids[best_model_index]
        if self.save_database:
            self.save_metrics()
            self.db_inserts.save_hyper_parameters_search(search_type, grid_search if self.grid_search_active else random_search, hp_id)

    def compute_ensembles(self, x_attack, plaintext_attack, ciphertext_attack, number_of_best_models):

        number_of_models = len(self.ge_all_validation)

        nt_interval = int(self.key_rank_attack_traces / self.key_rank_report_interval)

        list_of_best_models = ScaFunctions().get_best_models(number_of_models, self.ge_all_validation, nt_interval)

        self.ge_best_model_validation = self.ge_all_validation[list_of_best_models[0]]
        self.ge_best_model_attack = self.ge_all_attack[list_of_best_models[0]]
        self.sr_best_model_validation = self.sr_all_validation[list_of_best_models[0]]
        self.sr_best_model_attack = self.sr_all_attack[list_of_best_models[0]]

        kr_ensemble = np.zeros(nt_interval)
        krs_ensemble = np.zeros((self.key_rank_executions, nt_interval))
        kr_ensemble_best_models = np.zeros(nt_interval)
        krs_ensemble_best_models = np.zeros((self.key_rank_executions, nt_interval))

        nt = len(x_attack)
        labels_key_hypothesis = np.zeros((256, nt))
        for key_byte_hypothesis in range(0, 256):
            key_h = bytearray.fromhex(self.target_params["key"])
            key_h[self.leakage_model["byte"]] = key_byte_hypothesis
            labels_key_hypothesis[key_byte_hypothesis][:] = aes_intermediates_sr_ge(plaintext_attack, ciphertext_attack, key_h,
                                                                                    self.leakage_model)

        probabilities_kg_all_traces = np.zeros((number_of_models, nt, 256))

        for model_index in range(number_of_models):

            out_prob_model = self.output_probabilities_all_models[list_of_best_models[model_index]]
            for index in range(self.key_rank_attack_traces):
                probabilities_kg_all_traces[model_index][index] = out_prob_model[index][
                    np.asarray([int(leakage[index]) for leakage in labels_key_hypothesis[:]])
                    # array with 256 leakage values (1 per key guess)
                ]
            print("Processing Model {} of {}".format(model_index + 1, number_of_models))

        for run in range(self.key_rank_executions):

            key_p_ensemble = np.zeros(256)
            key_p_ensemble_best_models = np.zeros(256)

            probabilities_kg_all_traces_shuffled = np.zeros((number_of_models, nt, 256))
            for model_index in range(number_of_models):
                probabilities_kg_all_traces_shuffled[model_index] = shuffle(probabilities_kg_all_traces[model_index],
                                                                            random_state=random.randint(0, 100000))

            kr_count = 0
            for index in range(self.key_rank_attack_traces):
                for model_index in range(number_of_models):
                    key_p_ensemble += np.log(probabilities_kg_all_traces_shuffled[model_index][index] + 1e-36)
                for model_index in range(number_of_best_models):
                    key_p_ensemble_best_models += np.log(probabilities_kg_all_traces_shuffled[model_index][index] + 1e-36)

                key_p_ensemble_sorted = np.argsort(key_p_ensemble)[::-1]
                key_p_ensemble_best_models_sorted = np.argsort(key_p_ensemble_best_models)[::-1]

                if (index + 1) % self.key_rank_report_interval == 0:
                    kr_position = list(key_p_ensemble_sorted).index(self.target_params["good_key"]) + 1
                    kr_ensemble[kr_count] += kr_position
                    krs_ensemble[run][kr_count] = kr_position

                    kr_position = list(key_p_ensemble_best_models_sorted).index(self.target_params["good_key"]) + 1
                    kr_ensemble_best_models[kr_count] += kr_position
                    krs_ensemble_best_models[run][kr_count] = kr_position
                    kr_count += 1

            print("Run: {} | GE {} models: {} | GE {} best models: {}".format(run, number_of_models,
                                                                              kr_ensemble[nt_interval - 1] / (run + 1),
                                                                              number_of_best_models,
                                                                              kr_ensemble_best_models[nt_interval - 1] / (run + 1)))

        ge_ensemble = kr_ensemble / self.key_rank_executions
        ge_ensemble_best_models = kr_ensemble_best_models / self.key_rank_executions

        sr_ensemble = np.zeros(nt_interval)
        sr_ensemble_best_models = np.zeros(nt_interval)

        for index in range(nt_interval):
            for run in range(self.key_rank_executions):
                sr_ensemble[index] += 1 if krs_ensemble[run][index] == 1 else 0
                sr_ensemble_best_models[index] += 1 if krs_ensemble_best_models[run][index] == 1 else 0

        return ge_ensemble, ge_ensemble_best_models, sr_ensemble / self.key_rank_executions, \
               sr_ensemble_best_models / self.key_rank_executions, list_of_best_models

    def run(self, key_rank_executions=100, key_rank_report_interval=10, key_rank_attack_traces=1000, visualization=None,
            data_augmentation=None, ensemble=None, grid_search=None, random_search=None, early_stopping=None, confusion_matrix=False,
            callbacks=None, save_database=True, compute_ge=True, save_to_npz=None, probability_rank_plot=False):
        """

        Main AISY framework function. This function runs neural network training for profiled side-channel analysis in a known-key setting.

        :param key_rank_executions: number of times the key rank is computed for guessing entropy (averaged key rank)
        :param key_rank_report_interval: interval trace step to report metrics (key rank, guessing entropy, success rate, etc)
        :param key_rank_attack_traces:  number of attack traces randomly select from attack set in each key rank computation
        :param visualization: attributes for visualization feature (input gradients). Ex.: visualization=[4000]
        :param data_augmentation: attributes for data augmentation feature. Ex.: data_augmentation=[da_method_name, 400]
        :param ensemble: attributes for ensemble feature. Ex.: ensemble=[10]
        :param grid_search: dictionary containing definitions for grid search feature
        :param random_search: dictionary containing definitions for random search feature
        :param early_stopping: dictionary containing definitions for early stopping feature
        :param confusion_matrix: boolean variable to set the computation of confusion matrix
        :param callbacks: list of custom callbacks
        :param save_database: boolean variable setting database feature
        :param compute_ge: boolean variable setting guessing entropy feature
        :param save_to_npz: .npz file name to where results are saved (file results are placed in 'resources/npz' folder)
        :param probability_rank_plot: create probability rank plot according to https://tches.iacr.org/index.php/TCHES/article/view/8686)
        :return: None
        """

        for folder in ["databases", "figures", "models", "npz"]:
            dir_resources_id = "../resources/{}/".format(folder)
            if not os.path.exists(dir_resources_id):
                os.makedirs(dir_resources_id)

        if self.model is None and grid_search is None and random_search is None:
            print("ERROR 1: neural network model is not defined.")
            return

        if "filename" not in self.target_params.keys():
            print("ERROR 2: Dataset 'filename' not specified. Please use set_filename method to specify it.")
            return

        if "key" not in self.target_params.keys():
            print("ERROR 3: Dataset 'key' not specified. Please use set_key method to specify it.")
            return

        if "first_sample" not in self.target_params.keys():
            print("ERROR 4 : Dataset 'first_sample' not specified. Please use set_first_sample method to specify it.")
            return

        if "number_of_samples" not in self.target_params.keys():
            print("ERROR 5: Dataset 'number_of_samples' not specified. Please use set_number_of_samples method to specify it.")
            return

        if "number_of_profiling_traces" not in self.target_params.keys():
            print(
                "ERROR 6: Dataset 'number_of_profiling_traces' not specified. Please use set_number_of_profiling_traces method to specify it.")
            return

        if "number_of_attack_traces" not in self.target_params.keys():
            print("ERROR 7: Dataset 'number_of_attack_traces' not specified. Please use set_number_of_attack_traces method to specify it.")
            return

        # initialize configurations
        self.hyper_parameters = []
        self.key_rank_executions = key_rank_executions
        self.key_rank_attack_traces = key_rank_attack_traces
        self.key_rank_report_interval = key_rank_report_interval
        if visualization is not None:
            self.visualization_active = True
        if data_augmentation is not None:
            self.data_augmentation_active = True
        if ensemble is not None:
            self.ensemble_active = True
            self.ge_all_validation = []
            self.sr_all_validation = []
            self.ge_all_attack = []
            self.sr_all_attack = []
            self.output_probabilities_all_models = []
            if self.target_params["number_of_attack_traces"] == self.key_rank_attack_traces:
                print("ERROR 8: ensemble feature requires the 'number_of_attack_traces' >= 2 x key_rank_attack_traces.")
                return
        if grid_search is not None:
            self.grid_search_active = True
            if self.ensemble_active and grid_search["stop_condition"]:
                print("ERROR 11: when grid search has a stop condition, ensembles can't be applied.")
                return
            if self.target_params["number_of_attack_traces"] == self.key_rank_attack_traces:
                print("ERROR 14: grid search feature requires the 'number_of_attack_traces' >= 2 x key_rank_attack_traces.")
                return
        if random_search is not None:
            self.random_search_active = True
            if self.ensemble_active:
                if ensemble[0] > random_search["max_trials"]:
                    print("ERROR 12: number of models for ensembles can't be larger than maximum number of trials in random search.")
                    return
            if self.ensemble_active and random_search["stop_condition"]:
                print("ERROR 13: when random search has a stop condition, ensembles can't be applied.")
                return
            if self.target_params["number_of_attack_traces"] == self.key_rank_attack_traces:
                print("ERROR 15: random search feature requires the 'number_of_attack_traces' >= 2 x key_rank_attack_traces.")
                return
        if early_stopping is not None:
            self.early_stopping_active = True
            self.early_stopping_metrics = early_stopping["metrics"]
            if self.target_params["number_of_attack_traces"] == self.key_rank_attack_traces:
                print("ERROR 9: early stopping feature requires the 'number_of_attack_traces' >= 2 x key_rank_attack_traces.")
                return
        if confusion_matrix:
            self.confusion_matrix_active = True
        if save_to_npz is not None:
            self.save_to_npz = True
        if probability_rank_plot:
            self.probability_rank_plot = True
            self.output_probabilities = []
        self.compute_ge_active = compute_ge

        self.save_database = save_database

        nt_key_rank = int(self.key_rank_attack_traces / key_rank_report_interval)
        self.initialize_result_vectors(nt_key_rank)

        self.settings = {
            "key_rank_report_interval": self.key_rank_report_interval,
            "key_rank_attack_traces": self.key_rank_attack_traces,
            "key_rank_executions": self.key_rank_executions
        }
        if self.visualization_active:
            self.settings["visualization"] = visualization[0]
        # if self.data_augmentation_active:
        # self.settings["data_augmentation"] = [data_augmentation[0], data_augmentation[1]]
        if self.ensemble_active:
            self.settings["ensemble"] = ensemble[0]
        if self.grid_search_active:
            self.settings["grid_search"] = grid_search
        if self.random_search_active:
            self.settings["random_search"] = random_search
        if self.confusion_matrix_active:
            self.settings["confusion_matrix"] = True
        if self.early_stopping_active:
            self.settings["early_stopping"] = early_stopping
        # if callbacks is not None:
        #     self.settings["callbacks"] = callbacks
        if save_to_npz is not None:
            self.settings["save_to_npz"] = True

        if self.save_database:
            self.insert_new_analysis_in_database()

        # load data sets
        if ".h5" in self.target_params["filename"]:
            (X_profiling, Y_profiling), (X_validation, Y_validation), (X_attack, Y_attack), (
                plaintext_profiling, plaintext_validation, plaintext_attack,
                ciphertext_profiling, ciphertext_validation, ciphertext_attack,
                key_profiling, key_validation, key_attack) = ScaLoadDatasets().load_h5_dataset(
                self.datasets_root_folder + self.target_params["filename"],
                self.target_params, self.leakage_model,
                split_test_set=self.ensemble_active or self.early_stopping_active or self.grid_search_active or self.random_search_active)
        else:
            print("ERROR 10: Dataset format not supported.")
            return

        # normalize with z-score
        self.create_z_score_norm(X_profiling)
        self.apply_z_score_norm(X_profiling)
        if self.ensemble_active or self.early_stopping_active or self.grid_search_active or self.random_search_active:
            self.apply_z_score_norm(X_validation)
        self.apply_z_score_norm(X_attack)

        x_profiling = X_profiling.astype('float32')
        x_validation = X_validation.astype(
            'float32') if self.ensemble_active or self.early_stopping_active or self.grid_search_active or self.random_search_active else None
        x_attack = X_attack.astype('float32')

        # convert labels to categorical labels
        y_profiling = to_categorical(Y_profiling, num_classes=self.classes)
        if self.ensemble_active or self.early_stopping_active or self.grid_search_active or self.random_search_active:
            y_validation = to_categorical(Y_validation, num_classes=self.classes)
        else:
            y_validation = None
        y_attack = to_categorical(Y_attack, num_classes=self.classes)

        self.target_params["good_key"] = bytearray.fromhex(self.target_params["key"])[self.leakage_model["byte"]]

        self.start = time.time()

        if self.grid_search_active or self.random_search_active:
            self.run_search(x_profiling, y_profiling, x_attack, y_attack,
                            plaintext_profiling, plaintext_attack,
                            ciphertext_profiling, ciphertext_attack,
                            key_profiling, key_attack,
                            data_augmentation, visualization, self.key_rank_report_interval, self.key_rank_attack_traces,
                            grid_search, random_search,
                            x_validation=x_validation,
                            y_validation=y_validation,
                            plaintext_validation=plaintext_validation,
                            ciphertext_validation=ciphertext_validation,
                            key_validation=key_validation,
                            custom_callbacks=callbacks)
            if self.ensemble_active:
                ge_ensemble, ge_ensemble_best_models, sr_ensemble, sr_ensemble_best_models, list_of_best_models = self.compute_ensembles(
                    x_attack, plaintext_attack, ciphertext_attack, ensemble[0])

                self.ge_ensemble = ge_ensemble
                self.ge_ensemble_best_models = ge_ensemble_best_models
                self.sr_ensemble = sr_ensemble
                self.sr_ensemble_best_models = sr_ensemble_best_models

                if self.probability_rank_plot:
                    output_probabilities = np.zeros((len(x_attack), self.classes))
                    for model_index in range(ensemble[0]):
                        output_probabilities += self.output_probabilities[list_of_best_models[model_index]]
                    output_probabilities /= ensemble[0]
                    rank_results, _ = ScaFunctions().get_probability_ranks(x_attack, plaintext_attack, ciphertext_attack,
                                                                           self.key_rank_attack_traces,
                                                                           self.classes, self.leakage_model,
                                                                           self.target_params, self.model,
                                                                           output_probabilities=output_probabilities)
                    self.output_probabilities.append(output_probabilities)
                    if self.save_database:
                        self.__save_probability_ranks(rank_results, "Attack Byte {} Model Ensemble".format(self.leakage_model["byte"]))
                    self.save_probability_ranks_figure(rank_results, "Attack Byte {} Model Ensemble".format(self.leakage_model["byte"]))

                if self.save_database:
                    self.save_ensemble_results()
        else:
            self.train_model(x_profiling, y_profiling, x_attack, y_attack,
                             plaintext_profiling, plaintext_attack,
                             ciphertext_profiling, ciphertext_attack,
                             key_profiling, key_attack,
                             data_augmentation, visualization,
                             self.key_rank_report_interval, self.key_rank_attack_traces,
                             x_validation=x_validation,
                             plaintext_validation=plaintext_validation,
                             ciphertext_validation=ciphertext_validation,
                             key_validation=key_validation,
                             y_validation=y_validation,
                             custom_callbacks=callbacks)
            if self.save_database:
                self.learning_rate = backend.eval(self.model.optimizer.lr)
                self.optimizer = self.model.optimizer.__class__.__name__
                self.set_hyper_parameters(self.ge_attack[nt_key_rank - 1])
                self.save_results_in_database(time.time() - self.start, self.model_name)
                self.save_visualization_results()
                self.save_confusion_matrix_results()
                self.save_metrics()
                self.save_results()
                if self.early_stopping_active:
                    self.save_early_stopping_results()
            if self.save_to_npz:
                np.savez("../resources/npz/{}.npz".format(save_to_npz[0]),
                         metrics_profiling=self.metric_profiling,
                         metrics_validation=self.metric_validation,
                         metrics_attack=self.metric_attack,
                         guessing_entropy=self.ge_attack,
                         success_rate=self.sr_attack,
                         model_weights=self.model.get_weights(),
                         input_gradients_epoch=self.callback_input_gradients.grads_epoch() if self.visualization_active else None,
                         input_gradients_sum=self.callback_input_gradients.grads() if self.visualization_active else None,
                         settings=self.settings,
                         hyperparameters=self.hyper_parameters,
                         leakage_model=self.leakage_model,
                         model_description=ScaKerasModels().keras_model_as_string(self.model_class, self.model_name), allow_pickle=True
                         )
            if self.probability_rank_plot:
                rank_results, _ = ScaFunctions().get_probability_ranks(x_attack, plaintext_attack, ciphertext_attack,
                                                                       self.key_rank_attack_traces,
                                                                       self.classes, self.leakage_model,
                                                                       self.target_params, self.model)
                if self.save_database:
                    self.__save_probability_ranks(rank_results, "Attack Byte {}".format(self.leakage_model["byte"]))
                self.save_probability_ranks_figure(rank_results, "Attack Byte {}".format(self.leakage_model["byte"]))

    def __save_metric_avg(self, metric, n_models, name):
        kr_avg = sum(metric[n] for n in range(n_models)) / n_models
        self.db_inserts.save_metric(kr_avg, self.leakage_model["byte"], name)

    def __save_metric(self, metric, name):
        self.db_inserts.save_metric(metric, self.leakage_model["byte"], name)

    def __save_kr(self, kr, metric):
        self.db_inserts.save_key_rank_json(pd.Series(kr).to_json(), self.leakage_model["byte"], self.key_rank_report_interval, metric)

    def __save_sr(self, sr, metric):
        self.db_inserts.save_success_rate_json(pd.Series(sr).to_json(), self.leakage_model["byte"], self.key_rank_report_interval, metric)

    def __save_probability_ranks(self, ranks, title):
        for key_guess in range(256):
            self.db_inserts.save_probability_rank(pd.Series(ranks[key_guess]).to_json(), self.classes, self.target_params["good_key"],
                                                  key_guess, title, self.leakage_model["byte"])

    def save_probability_ranks_figure(self, ranks, title):
        my_dpi = 100
        import matplotlib.pyplot as plt
        plt.figure(figsize=(800 / my_dpi, 600 / my_dpi), dpi=my_dpi)

        for kg in range(256):
            if kg != self.target_params["good_key"]:
                if kg == 0:
                    plt.plot(np.arange(1, self.classes + 1), ranks[kg], color="#bdbdbd", label="Wrong Key Hypotheses")
                else:
                    plt.plot(np.arange(1, self.classes + 1), ranks[kg], color="#bdbdbd")
        plt.plot(np.arange(1, self.classes + 1), ranks[self.target_params["good_key"]], label="Correct Key Hypothesis", color="purple")
        plt.legend(loc='best', fontsize=13)
        plt.title(title)
        plt.xlabel("Class Probability Rank", fontsize=13)
        plt.ylabel("Density", fontsize=13)
        plt.grid(ls='--')
        plt.xlim([1, self.classes])
        timestamp = str(time.time()).replace(".", "")
        dir_analysis_id = "../resources/figures/{}".format(self.db_inserts.get_analysis_id())
        analysis_id = self.db_inserts.get_analysis_id()
        if not os.path.exists(dir_analysis_id):
            os.makedirs(dir_analysis_id)
        plt.savefig("../resources/figures/{}/probability_ranks_{}_{}.png".format(analysis_id, title.replace(" ", "_"), timestamp),
                    format="png")

    def set_hyper_parameters(self, ge):
        self.hyper_parameters.append({
            "GE": ge,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": float(self.learning_rate),
            "optimizer": str(self.optimizer),
            "key": self.target_params["key"],
            "profiling_traces": self.target_params["number_of_profiling_traces"],
            "attack_traces": self.target_params["number_of_attack_traces"],
            "first_sample": self.target_params["first_sample"],
            "number_of_samples": self.target_params["number_of_samples"]
        })
        self.hyper_parameters[0]["parameters"] = self.model.count_params()

    def set_hyper_parameters_search(self, hp, ge):
        self.hyper_parameters_search.append({
            "GE": ge,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "optimizer": str(self.optimizer),
            "key": self.target_params["key"],
            "profiling_traces": self.target_params["number_of_profiling_traces"],
            "attack_traces": self.target_params["number_of_attack_traces"],
            "first_sample": self.target_params["first_sample"],
            "number_of_samples": self.target_params["number_of_samples"]
        })
        for key, value in hp.items():
            self.hyper_parameters_search[0][key] = value
        self.hyper_parameters_search[0]["parameters"] = self.model.count_params()

    def insert_new_analysis_in_database(self):
        self.db_inserts = ScaDatabaseInserts(self.database_root_folder + self.database_name, self.database_name,
                                             self.target_params["filename"], self.settings, 0)

    def save_results_in_database(self, elapsed_time, model_name, update=False, hyperparameters_search=False):
        if update:
            self.db_inserts.update_elapsed_time_analysis(elapsed_time)
            if hyperparameters_search:
                return self.db_inserts.save_hyper_parameters(self.hyper_parameters_search)
            else:
                return self.db_inserts.save_hyper_parameters(self.hyper_parameters)
        else:
            self.db_inserts.update_elapsed_time_analysis(elapsed_time)

            leakage_model = [{
                "cipher": self.leakage_model["cipher"],
                "leakage_model": self.leakage_model["leakage_model"],
                "byte": self.leakage_model["byte"],
                "round": self.leakage_model["round"],
                "target_state": self.leakage_model["target_state"],
                "direction": self.leakage_model["direction"],
                "attack_direction": self.leakage_model["attack_direction"]
            }]
            if self.leakage_model["leakage_model"] == "bit":
                leakage_model[0]["bit"] = self.leakage_model["bit"]

            if self.leakage_model["leakage_model"] == "HD":
                leakage_model[0]["round_first"] = self.leakage_model["round_first"]
                leakage_model[0]["round_second"] = self.leakage_model["round_second"]
                leakage_model[0]["target_state_first"] = self.leakage_model["target_state_first"]
                leakage_model[0]["target_state_second"] = self.leakage_model["target_state_second"]

            sca_keras_model = ScaKerasModels()
            if hyperparameters_search:
                model_description = "Check the list of searched hyper-parameters"
            else:
                model_description = sca_keras_model.keras_model_as_string(self.model_class, model_name)

            self.db_inserts.save_neural_network(model_description, model_name)
            self.db_inserts.save_leakage_model(leakage_model)
            if hyperparameters_search:
                return self.db_inserts.save_hyper_parameters(self.hyper_parameters_search)
            else:
                return self.db_inserts.save_hyper_parameters(self.hyper_parameters)

    def save_metrics(self):
        for metric_profiling in self.metric_profiling:
            self.__save_metric(metric_profiling["values"], metric_profiling["metric"])
        for metric_validation in self.metric_validation:
            self.__save_metric(metric_validation["values"], metric_validation["metric"])
        for metric_attack in self.metric_attack:
            self.__save_metric(metric_attack["values"], metric_attack["metric"])

    def save_results(self, best_model_search=False, search_index=None):
        is_validation = (self.ensemble_active or self.early_stopping_active or self.grid_search_active or self.random_search_active) \
                        and not best_model_search
        if best_model_search and not self.ensemble_active:
            db_metric_name = "Validation Set Best Model" if is_validation else "Attack Set Best Model"
        elif search_index is not None:
            db_metric_name = "Validation Set {}".format(search_index) if is_validation else "Attack Set {}".format(search_index)
        else:
            db_metric_name = "Validation Set" if is_validation else "Attack Set"
        if search_index is not None or not self.ensemble_active:
            self.__save_kr(self.ge_validation if is_validation else self.ge_attack, db_metric_name)
            self.__save_sr(self.sr_validation if is_validation else self.sr_attack, db_metric_name)

    def save_ensemble_results(self):
        self.__save_kr(self.ge_best_model_attack, "Best Model Attack")
        self.__save_kr(self.ge_best_model_validation, "Best Model Validation")
        self.__save_kr(self.ge_ensemble, "Ensemble")
        self.__save_kr(self.ge_ensemble_best_models, "Ensemble Best Models")
        self.__save_sr(self.sr_best_model_attack, "Best Model Attack")
        self.__save_sr(self.sr_best_model_validation, "Best Model Validation")
        self.__save_sr(self.sr_ensemble, "Ensemble")
        self.__save_sr(self.sr_ensemble_best_models, "Ensemble Best Models")

    def save_early_stopping_results(self, best_model_search=False, search_index=None):
        for ge_attack_early_stopping in self.ge_attack_early_stopping:
            if best_model_search:
                db_metric_name = "ES {} Best Model".format(ge_attack_early_stopping["metric"])
            elif search_index is not None:
                db_metric_name = "ES {} {}".format(ge_attack_early_stopping["metric"], search_index)
            else:
                db_metric_name = "ES {}".format(ge_attack_early_stopping["metric"])
            self.__save_kr(ge_attack_early_stopping["guessing_entropy"], db_metric_name)
        for sr_attack_early_stopping in self.sr_attack_early_stopping:
            if best_model_search:
                db_metric_name = "ES {} Best Model".format(sr_attack_early_stopping["metric"])
            elif search_index is not None:
                db_metric_name = "ES {} {}".format(sr_attack_early_stopping["metric"], search_index)
            else:
                db_metric_name = "ES {}".format(sr_attack_early_stopping["metric"])
            self.__save_sr(sr_attack_early_stopping["success_rate"], db_metric_name)

    def save_visualization_results(self):
        if self.visualization_active:
            input_gradients_epoch = self.callback_input_gradients.grads_epoch()
            for epoch in range(self.epochs):
                self.db_inserts.save_visualization(pd.Series(input_gradients_epoch[epoch]).to_json(), epoch, self.leakage_model["byte"],
                                                   self.key_rank_report_interval, "InputGradient")
            input_gradients_sum = self.callback_input_gradients.grads()
            self.db_inserts.save_visualization(pd.Series(input_gradients_sum / self.epochs).to_json(),
                                               self.epochs, self.leakage_model["byte"], self.key_rank_report_interval, "InputGradient")

    def save_confusion_matrix_results(self):
        if self.confusion_matrix_active:
            cm = self.callback_confusion_matrix.get_confusion_matrix()
            for y_true, y_pred in enumerate(cm):
                self.db_inserts.save_confusion_matrix(pd.Series(y_pred).to_json(), y_true, self.leakage_model["byte"])

    def compute_ge_and_sr(self, x_attack, plaintext_attack, ciphertext_attack,
                          key_rank_report_interval, key_rank_attack_traces, x_validation=None,
                          plaintext_validation=None, ciphertext_validation=None, early_stopping_metric_results=None, best_model=False):
        if (self.grid_search_active or self.random_search_active or self.ensemble_active) and not best_model:
            print(colored("Computing Guessing Entropy and Success Rate for Validation Set", "blue"))
            self.ge_validation, self.sr_validation, _ = ScaFunctions().ge_and_sr(self.key_rank_executions, self.model,
                                                                                 self.target_params,
                                                                                 self.leakage_model,
                                                                                 x_validation, plaintext_validation,
                                                                                 ciphertext_validation,
                                                                                 key_rank_report_interval,
                                                                                 key_rank_attack_traces)
            if self.ensemble_active:
                print(colored("Computing Guessing Entropy and Success Rate for Attack Set", "green"))
                self.ge_attack, self.sr_attack, output_probabilities = ScaFunctions().ge_and_sr(self.key_rank_executions, self.model,
                                                                                                self.target_params,
                                                                                                self.leakage_model,
                                                                                                x_attack, plaintext_attack,
                                                                                                ciphertext_attack,
                                                                                                key_rank_report_interval,
                                                                                                key_rank_attack_traces)

                self.ge_all_validation.append(self.ge_validation)
                self.ge_all_attack.append(self.ge_attack)
                self.sr_all_validation.append(self.sr_validation)
                self.sr_all_attack.append(self.sr_attack)
                self.output_probabilities_all_models.append(output_probabilities)

        else:
            print(colored("Computing Guessing Entropy and Success Rate for Attack Set", "green"))
            self.ge_attack, self.sr_attack, output_probabilities = ScaFunctions().ge_and_sr(self.key_rank_executions, self.model,
                                                                                            self.target_params,
                                                                                            self.leakage_model,
                                                                                            x_attack, plaintext_attack, ciphertext_attack,
                                                                                            key_rank_report_interval,
                                                                                            key_rank_attack_traces)

        if self.early_stopping_active:
            self.ge_attack_early_stopping = []
            self.sr_attack_early_stopping = []
            for early_stopping_metric in self.early_stopping_metrics:
                if isinstance(early_stopping_metric_results[early_stopping_metric][0], list):
                    for i in range(len(early_stopping_metric_results[early_stopping_metric][0])):
                        print("../resources/models/best_model_{}_{}_{}.h5".format(early_stopping_metric, self.timestamp, i))
                        self.model.load_weights("../resources/models/best_model_{}_{}_{}.h5".format(early_stopping_metric,
                                                                                                    self.timestamp, i))
                        print(colored("Computing Guessing Entropy and Success Rate for Attack Set", "green"))
                        ge_attack, sr_attack, _ = ScaFunctions().ge_and_sr(self.key_rank_executions, self.model, self.target_params,
                                                                           self.leakage_model,
                                                                           x_attack, plaintext_attack, ciphertext_attack,
                                                                           key_rank_report_interval, key_rank_attack_traces)
                        self.ge_attack_early_stopping.append(
                            {
                                "metric": "{}_{}".format(early_stopping_metric, i),
                                "guessing_entropy": ge_attack
                            }
                        )

                        self.sr_attack_early_stopping.append(
                            {
                                "metric": "{}_{}".format(early_stopping_metric, i),
                                "success_rate": sr_attack
                            }
                        )
                else:
                    print("../resources/models/best_model_{}_{}.h5".format(early_stopping_metric, self.timestamp))
                    self.model.load_weights("../resources/models/best_model_{}_{}.h5".format(early_stopping_metric,
                                                                                             self.timestamp))
                    print(colored("Computing Guessing Entropy and Success Rate for Attack Set", "green"))
                    ge_attack, sr_attack, _ = ScaFunctions().ge_and_sr(self.key_rank_executions, self.model,
                                                                       self.target_params, self.leakage_model,
                                                                       x_attack, plaintext_attack, ciphertext_attack,
                                                                       key_rank_report_interval, key_rank_attack_traces)
                    self.ge_attack_early_stopping.append(
                        {
                            "metric": early_stopping_metric,
                            "guessing_entropy": ge_attack
                        }
                    )

                    self.sr_attack_early_stopping.append(
                        {
                            "metric": early_stopping_metric,
                            "success_rate": sr_attack
                        }
                    )

    def get_metrics_results(self, history, search_index, metric_results=None):
        self.metric_profiling.append(
            {
                "values": history.history["accuracy"],
                "metric": "accuracy_{}".format(search_index) if search_index is not None else "accuracy"
            })
        self.metric_profiling.append(
            {
                "values": history.history["loss"],
                "metric": "loss_{}".format(search_index) if search_index is not None else "loss"
            })

        if self.early_stopping_active:
            for metric in self.early_stopping_metrics:
                if isinstance(metric_results[metric][0], list):
                    for i in range(len(metric_results[metric][0])):
                        self.metric_validation.append(
                            {
                                "values": np.array(metric_results[metric])[:, i],
                                "metric": "val_{}_{} {}".format(metric, i,
                                                                search_index) if search_index is not None else "val_{}_{}".format(
                                    metric, i)
                            })
                else:
                    self.metric_validation.append(
                        {
                            "values": metric_results[metric],
                            "metric": "val_{} {}".format(metric, search_index) if search_index is not None else "val_{}".format(metric)
                        })
        else:
            self.metric_attack.append(
                {
                    "values": history.history["val_accuracy"],
                    "metric": "val_accuracy_{}".format(search_index) if search_index is not None else "val_accuracy"
                })
            self.metric_attack.append(
                {
                    "values": history.history["val_loss"],
                    "metric": "val_loss_{}".format(search_index) if search_index is not None else "val_loss"
                })
