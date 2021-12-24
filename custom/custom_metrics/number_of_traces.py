import random
from sklearn.utils import shuffle
from aisy_sca.crypto.sca_aes_create_intermediates import *

def run(dataset, settings, model, *args):

    nt = len(dataset.x_validation)
    nt_interval = int(settings["key_rank_attack_traces"] / settings["key_rank_report_interval"])
    key_ranking_sum = np.zeros(nt_interval)

    # ---------------------------------------------------------------------------------------------------------#
    # compute labels for key hypothesis
    # ---------------------------------------------------------------------------------------------------------#
    labels_key_hypothesis = np.zeros((256, nt))
    for key_byte_hypothesis in range(0, 256):
        key_h = bytearray.fromhex(settings["key"])
        key_h[settings["leakage_model"]["byte"]] = key_byte_hypothesis
        labels_key_hypothesis[key_byte_hypothesis][:] = aes_intermediates(dataset.plaintext_validation, dataset.ciphertext_validation,
                                                                                key_h, settings["leakage_model"])

    good_key = [int(x) for x in bytearray.fromhex(settings["key"])][settings["leakage_model"]["byte"]]

    # ---------------------------------------------------------------------------------------------------------#
    # predict output probabilities for shuffled test or validation set
    # ---------------------------------------------------------------------------------------------------------#
    output_probabilities = model.predict(dataset.x_validation)

    probabilities_kg_all_traces = np.zeros((nt, 256))
    for index in range(nt):
        probabilities_kg_all_traces[index] = output_probabilities[index][
            np.asarray([int(leakage[index]) for leakage in labels_key_hypothesis[:]])  # array with 256 leakage values (1 per key guess)
        ]

    for key_rank_execution in range(settings["key_rank_executions"]):

        probabilities_kg_all_traces_shuffled = shuffle(probabilities_kg_all_traces, random_state=random.randint(0, 100000))
        key_probabilities = np.zeros(256)

        kr_count = 0
        for index in range(settings["key_rank_attack_traces"]):

            key_probabilities += np.log(probabilities_kg_all_traces_shuffled[index] + 1e-36)
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]

            if (index + 1) % settings["key_rank_report_interval"] == 0:
                key_ranking_good_key = list(key_probabilities_sorted).index(good_key) + 1
                key_ranking_sum[kr_count] += key_ranking_good_key
                kr_count += 1

    guessing_entropy = key_ranking_sum / settings["key_rank_executions"]

    result_number_of_traces_val = settings["key_rank_attack_traces"]
    if np.floor(guessing_entropy[nt_interval - 1]) == 1:
        for index in range(nt_interval - 1, -1, -1):
            if np.floor(guessing_entropy[index]) != 1:
                result_number_of_traces_val = (index + 1) * settings["key_rank_report_interval"]
                break

    print("Number of traces to reach GE = 1: {}".format(result_number_of_traces_val))
    return result_number_of_traces_val
