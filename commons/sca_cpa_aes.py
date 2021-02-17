import numpy as np
import pandas as pd
import time
import os
import itertools
import json
from tensorflow.keras import backend
from tensorflow.keras.utils import to_categorical
from custom.custom_datasets.datasets import *
from commons.sca_callbacks import *
from commons.sca_keras_models import ScaKerasModels
from commons.sca_functions import ScaFunctions
from commons.sca_database_inserts import ScaDatabaseInserts
from commons.sca_data_augmentation import ScaDataAugmentation
from commons.sca_load_datasets import ScaLoadDatasets
from app import databases_root_folder, datasets_root_folder
from neural_networks.neural_networks_grid_search import *
from neural_networks.neural_networks_random_search import *
from termcolor import colored


class AisyCPA:

    def __init__(self):
        self.datasets_root_folder = datasets_root_folder
        self.database_root_folder = databases_root_folder
        self.target_params = {}
        self.leakage_model = None
        self.key_rank_report_interval = 10
        self.key_int = None
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

    def set_aes_leakage_model(self, leakage_model="HW", bit=0, byte=0, round=1, cipher="AES128", target_state="Sbox",
                              direction="Encryption", attack_direction="input"):

        """
        Function to set the AES Leakage Model in the profiled SCA execution.
        :parameter
            leakage_model: 'HW', 'ID' or 'bit'
            bit: index of target bit (min 0, max 7)
            byte: index of target key byte
            round: index the target round
            target_state: 'Sbox', InvSbox', 'AddRoundKey', 'MixColumns', 'InvMixColumns', 'ShiftRows', 'InvShiftRows'
            direction: 'Encryption', 'Decryption'

        :return
            dictionary containing AES leakage model information:

            self.leakage_model = {
                "leakage_model": leakage_model,
                "bit": bit,
                "byte": byte,
                "round": round,
                "target_state": target_state,
                "direction": direction,
                "attack_direction": input
            }

        """

        self.leakage_model = {
            "leakage_model": leakage_model,
            "bit": bit,
            "byte": byte,
            "round": 1,
            "cipher": cipher,
            "target_state": target_state,
            "direction": direction,
            "attack_direction": attack_direction
        }

        if self.target_params is not None:
            if self.leakage_model["leakage_model"] == "HW":
                self.classes = 9
            elif self.leakage_model["leakage_model"] == "ID":
                self.classes = 256
            else:
                self.classes = 2
        else:
            print("Parameters (param) from target is not selected. Set target before the leakage model.")

        return self.leakage_model

    def run(self, key_rank_report_interval=10):

        self.key_rank_report_interval = key_rank_report_interval

        # load data sets
        if ".h5" in self.target_params["filename"]:
            (X_profiling, Y_profiling), (X_validation, Y_validation), (X_attack, Y_attack), (
                plaintext_profiling, plaintext_validation, plaintext_attack,
                ciphertext_profiling, ciphertext_validation, ciphertext_attack,
                key_profiling, key_validation, key_attack) = ScaLoadDatasets().load_h5_dataset(
                self.datasets_root_folder + self.target_params["filename"],
                self.target_params, self.leakage_model, do_labeling=False)
        else:
            print("ERROR: Dataset format not supported.")
            return

        return ScaFunctions().data_correlation(X_profiling, plaintext_profiling, key_profiling, self.leakage_model,
                                               key_rank_report_interval)
