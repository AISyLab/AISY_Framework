import numpy as np
import random
from sklearn.utils import shuffle
from aisy.sca_aes_create_intermediates import *


class ScaFunctions:

    def ge_and_sr(self, runs, model, param, leakage_model, x_attack, plaintext_attack, ciphertext_attack,
                  key_rank_report_interval, key_rank_attack_traces):

        nt = len(x_attack)
        nt_interval = int(key_rank_attack_traces / key_rank_report_interval)
        key_ranking_sum = np.zeros(nt_interval)
        success_rate_sum = np.zeros(nt_interval)

        # ---------------------------------------------------------------------------------------------------------#
        # compute labels for key hypothesis
        # ---------------------------------------------------------------------------------------------------------#
        labels_key_hypothesis = np.zeros((256, nt))
        for key_byte_hypothesis in range(0, 256):
            key_h = bytearray.fromhex(param["key"])
            key_h[leakage_model["byte"]] = key_byte_hypothesis
            labels_key_hypothesis[key_byte_hypothesis][:] = aes_intermediates_sr_ge(plaintext_attack, ciphertext_attack, key_h,
                                                                                    leakage_model)

        # ---------------------------------------------------------------------------------------------------------#
        # predict output probabilities for shuffled test or validation set
        # ---------------------------------------------------------------------------------------------------------#
        output_probabilities = model.predict(x_attack)

        probabilities_kg_all_traces = np.zeros((nt, 256))
        for index in range(nt):
            probabilities_kg_all_traces[index] = output_probabilities[index][
                np.asarray([int(leakage[index]) for leakage in labels_key_hypothesis[:]])  # array with 256 leakage values (1 per key guess)
            ]

        for run in range(runs):

            probabilities_kg_all_traces_shuffled = shuffle(probabilities_kg_all_traces, random_state=random.randint(0, 100000))

            key_probabilities = np.zeros(256)

            kr_count = 0
            for index in range(key_rank_attack_traces):

                key_probabilities += np.log(probabilities_kg_all_traces_shuffled[index] + 1e-36)
                key_probabilities_sorted = np.argsort(key_probabilities)[::-1]

                if (index + 1) % key_rank_report_interval == 0:
                    key_ranking_good_key = list(key_probabilities_sorted).index(param["good_key"]) + 1
                    key_ranking_sum[kr_count] += key_ranking_good_key

                    if key_ranking_good_key == 1:
                        success_rate_sum[kr_count] += 1

                    kr_count += 1

            print("Computing GE - KR run: {} | final GE for correct key ({}): {})".format(run, param["good_key"],
                                                                                          key_ranking_sum[nt_interval - 1] / (run + 1)))

        guessing_entropy = key_ranking_sum / runs
        success_rate = success_rate_sum / runs

        return guessing_entropy, success_rate, output_probabilities

    def get_probability_ranks(self, x_attack, plaintext_attack, ciphertext_attack, key_rank_attack_traces, classes, leakage_model, param,
                              model, output_probabilities=None):

        if output_probabilities is None:
            output_probabilities = model.predict(x_attack)

        intermediates_key_hypothesis = np.zeros((256, key_rank_attack_traces))
        for key_guess in range(256):
            key_h = bytearray.fromhex(param["key"])
            key_h[leakage_model["byte"]] = key_guess
            intermediates_key_hypothesis[key_guess][:] = aes_intermediates_sr_ge(plaintext_attack, ciphertext_attack, key_h,
                                                                                 leakage_model)

        p = np.zeros((256, key_rank_attack_traces))

        for key_guess in range(256):
            for i in range(key_rank_attack_traces):
                p[key_guess][i] = int(list(np.argsort(output_probabilities[i])[::-1]).index(intermediates_key_hypothesis[key_guess][i]))

        probabilities_count = np.zeros((256, classes))

        for key_guess in range(256):
            for class_index in range(classes):
                probabilities_count[key_guess][class_index] += np.count_nonzero(p[key_guess] == class_index)
            probabilities_count[key_guess] /= key_rank_attack_traces

        return probabilities_count, output_probabilities

    def get_best_models(self, n_models, result_models_validation, n_traces):

        """
        Compute list of best models based on the GE.
        """

        result_number_of_traces_val = []
        for model_index in range(n_models):
            if result_models_validation[model_index][n_traces - 1] == 1:
                for index in range(n_traces - 1, -1, -1):
                    if result_models_validation[model_index][index] != 1:
                        result_number_of_traces_val.append(
                            [result_models_validation[model_index][n_traces - 1], index + 1,
                             model_index])
                        break
            else:
                result_number_of_traces_val.append(
                    [result_models_validation[model_index][n_traces - 1], n_traces,
                     model_index])

        sorted_models = sorted(result_number_of_traces_val, key=lambda l: l[:])

        list_of_best_models = []
        for model_index in range(n_models):
            list_of_best_models.append(sorted_models[model_index][2])

        return list_of_best_models
