import numpy as np
import h5py
from aisy.sca_aes_create_intermediates import *


class ScaLoadDatasets:

    def load_h5_dataset(self, dataset_file, params, leakage_model, split_test_set=False, do_labeling=True):

        n_profiling = params["number_of_profiling_traces"]
        n_attack = params["number_of_attack_traces"]

        in_file = h5py.File(dataset_file, "r")
        profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
        attack_samples = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
        profiling_plaintext = in_file['Profiling_traces/metadata']['plaintext']
        attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
        profiling_key = in_file['Profiling_traces/metadata']['key']
        attack_key = in_file['Attack_traces/metadata']['key']
        profiling_ciphertext = np.zeros((n_profiling, len(profiling_plaintext[0])))
        attack_ciphertext = np.zeros((n_attack, len(attack_plaintext[0])))
        if leakage_model["direction"] == "Decryption" or leakage_model["attack_direction"] == "output":
            profiling_ciphertext = in_file['Profiling_traces/metadata']['ciphertext']
            attack_ciphertext = in_file['Attack_traces/metadata']['ciphertext']

        nt = n_profiling
        na = n_attack

        X_profiling = profiling_samples[0:nt]
        X_attack = attack_samples[0:na]

        profiling_plaintext = profiling_plaintext[:nt]
        attack_plaintext = attack_plaintext[:na]

        profiling_ciphertext = profiling_ciphertext[:nt]
        attack_ciphertext = attack_ciphertext[:na]

        profiling_key = profiling_key[:nt]
        attack_key = attack_key[:na]

        if do_labeling:
            if leakage_model["cipher"] == "AES128":
                Y_profiling = aes_intermediates(profiling_plaintext, profiling_ciphertext, profiling_key, leakage_model)
                Y_attack = aes_intermediates(attack_plaintext, attack_ciphertext, attack_key, leakage_model)
            else:
                print("ERROR: cipher not supported.")
                return
        else:
            Y_profiling = None
            Y_attack = None

        # attack set is split into validation and attack sets.
        X_validation = None
        Y_validation = None
        validation_plaintext = None
        validation_ciphertext = None
        validation_key = None
        if split_test_set:
            X_validation = X_attack[0: int(na / 2)]
            Y_validation = Y_attack[0: int(na / 2)]
            X_attack = X_attack[int(na / 2): na]
            Y_attack = Y_attack[int(na / 2): na]

            profiling_plaintext = profiling_plaintext[0:nt]
            validation_plaintext = attack_plaintext[0: int(na / 2)]
            attack_plaintext = attack_plaintext[int(na / 2): na]

            profiling_ciphertext = profiling_ciphertext[0:nt]
            validation_ciphertext = attack_ciphertext[0: int(na / 2)]
            attack_ciphertext = attack_ciphertext[int(na / 2): na]

            profiling_key = profiling_key[0:nt]
            validation_key = attack_key[0: int(na / 2)]
            attack_key = attack_key[int(na / 2): na]

        return (X_profiling, Y_profiling), (X_validation, Y_validation), (X_attack, Y_attack), (
            profiling_plaintext, validation_plaintext, attack_plaintext,
            profiling_ciphertext, validation_ciphertext, attack_ciphertext,
            profiling_key, validation_key, attack_key)
