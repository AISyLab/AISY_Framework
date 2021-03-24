import numpy as np
import random
from sklearn.utils import shuffle
from aisy_sca.sca_aes_create_intermediates import *


def run(x_profiling, y_profiling, plaintexts_profiling,
        ciphertexts_profiling, key_profiling,
        x_validation, y_validation, plaintexts_validation,
        ciphertexts_validation, key_validation,
        x_attack, y_attack, plaintexts_attack,
        ciphertexts_attack, key_attack,
        param, leakage_model,
        key_rank_executions, key_rank_report_interval, key_rank_attack_traces,
        model, *args):

    nt = len(x_validation)
    nt_interval = int(key_rank_attack_traces / key_rank_report_interval)
    success_rate_sum = np.zeros(nt_interval)

    # ---------------------------------------------------------------------------------------------------------#
    # compute labels for key hypothesis
    # ---------------------------------------------------------------------------------------------------------#
    labels_key_hypothesis = np.zeros((256, nt))
    for key_byte_hypothesis in range(0, 256):
        key_h = bytearray.fromhex(param["key"])
        key_h[leakage_model["byte"]] = key_byte_hypothesis
        labels_key_hypothesis[key_byte_hypothesis][:] = aes_intermediates_sr_ge(plaintexts_validation, ciphertexts_validation, key_h,
                                                                                leakage_model)
    good_key = [int(x) for x in bytearray.fromhex(param["key"])][leakage_model["byte"]]

    # ---------------------------------------------------------------------------------------------------------#
    # predict output probabilities for shuffled test or validation set
    # ---------------------------------------------------------------------------------------------------------#
    output_probabilities = model.predict(x_validation)

    probabilities_kg_all_traces = np.zeros((nt, 256))
    for index in range(nt):
        probabilities_kg_all_traces[index] = output_probabilities[index][
            np.asarray([int(leakage[index]) for leakage in labels_key_hypothesis[:]])  # array with 256 leakage values (1 per key guess)
        ]

    for key_rank_execution in range(key_rank_executions):

        probabilities_kg_all_traces_shuffled = shuffle(probabilities_kg_all_traces, random_state=random.randint(0, 100000))
        key_probabilities = np.zeros(256)

        kr_count = 0
        for index in range(key_rank_attack_traces):

            key_probabilities += np.log(probabilities_kg_all_traces_shuffled[index] + 1e-36)
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]

            if (index + 1) % key_rank_report_interval == 0:
                key_ranking_good_key = list(key_probabilities_sorted).index(good_key) + 1

                if key_ranking_good_key == 1:
                    success_rate_sum[kr_count] += 1

                kr_count += 1

        final_sr = success_rate_sum[nt_interval - 1]
        print("KR run: {} | final Success Rate for correct key ({}): {})".format(key_rank_execution + 1, good_key,
                                                                                 final_sr / (key_rank_execution + 1)))

    success_rate = success_rate_sum / key_rank_executions

    return success_rate[nt_interval - 1]
