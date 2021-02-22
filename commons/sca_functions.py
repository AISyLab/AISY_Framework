import numpy as np
import random
from sklearn.utils import shuffle
from commons.sca_aes_create_intermediates import *


class ConditionalAveragerAes:

    def __init__(self, numValues, traceLength):
        # Allocate the matrix of averaged traces
        self.avtraces = np.zeros((numValues, traceLength))
        self.counters = np.zeros(numValues)
        # print("ConditionalAverager: initialized for {} values and trace length {}".format(numValues, traceLength))

    def addTrace(self, data, trace):
        # Add a single trace with corresponding single chunk of data
        data = int(data)
        if self.counters[data] == 0:
            self.avtraces[data] = trace
        else:
            self.avtraces[data] = self.avtraces[data] + (trace - self.avtraces[data]) / self.counters[data]
        self.counters[data] += 1

    def getSnapshot(self):
        # return a snapshot of the average matrix'''
        avdataSnap = np.flatnonzero(self.counters)  # get an vector of only _observed_ values
        avtracesSnap = self.avtraces[avdataSnap]  # remove lines corresponding to non-observed values
        return avdataSnap, avtracesSnap


class ScaFunctions:

    def ge_and_sr(self, runs, model, param, leakage_model, x_attack, plaintext_attack, ciphertext_attack,
                  key_rank_report_interval, key_rank_attack_traces):

        nt = len(x_attack)
        nt_interval = int(key_rank_attack_traces / key_rank_report_interval)
        key_ranking_sum = np.zeros(nt_interval)
        success_rate_sum = np.zeros(nt_interval)
        key_probabilities_key_ranks = np.zeros((runs, nt, 256))

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
                key_probabilities_key_ranks[run][index] = probabilities_kg_all_traces_shuffled[index]
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

        return guessing_entropy, success_rate, key_probabilities_key_ranks

    def get_probability_ranks(self, x_attack, plaintext_attack, ciphertext_attack, key_rank_attack_traces, key_rank_executions,
                              classes, leakage_model, param, model):

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

        return probabilities_count

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

    # Even faster correlation trace computation
    # Takes the full matrix of predictions instead of just a column
    # O - (n,t) array of n traces with t samples each
    # P - (n,m) array of n predictions for each of the m candidates
    # returns an (m,t) correlation matrix of m traces t samples each
    def correlation_traces(self, o, p):
        n, t = o.shape  # n traces of t samples
        # (n_bis, m) = P.shape  # n predictions for each of m candidates

        do = o - (np.einsum("nt->t", o, dtype='float64', optimize='optimal') / np.double(n))  # compute O - mean(O)
        dp = p - (np.einsum("nm->m", p, dtype='float64', optimize='optimal') / np.double(n))  # compute P - mean(P)

        numerator = np.einsum("nm,nt->mt", dp, do, optimize='optimal')
        tmp1 = np.einsum("nm,nm->m", dp, dp, optimize='optimal')
        tmp2 = np.einsum("nt,nt->t", do, do, optimize='optimal')
        tmp = np.einsum("m,t->mt", tmp1, tmp2, optimize='optimal')
        denominator = np.sqrt(tmp)

        return numerator / denominator

    def sBoxOut(self, data, keyByte):
        sBoxIn = data ^ keyByte
        return self.sbox[sBoxIn]

    def cpa(self, data, samples, correct_key, leakage_model):
        h = np.zeros((1, len(data)), dtype='uint8')  # intermediate variable predictions
        hl = np.zeros((1, len(data)))

        if leakage_model["leakage_model"] == "HW":
            h[0, :] = self.sBoxOut(data, correct_key)
            hl[0, :] = [bin(iv).count("1") for iv in h[0, :]]
        elif leakage_model["leakage_model"] == "ID":
            hl[0, :] = self.sBoxOut(data, correct_key)
        else:
            h[0, :] = self.sBoxOut(data, correct_key)
            hl[0, :] = [int(bin(iv >> leakage_model["bit"])[len(bin(iv >> leakage_model["bit"])) - 1]) for iv in h[0, :]]

        hl = np.array(hl).T
        corr_traces = self.correlation_traces(samples, hl)

        return corr_traces

    def data_correlation(self, samples, plaintext, key, leakage_model, report_interval):

        # todo: support for different leakage models in AES (only S-Box output, round 1 for now)

        number_of_traces = len(samples)
        number_of_points = len(samples[0])

        labels = plaintext[:, leakage_model["byte"]]

        corr_traces = None

        tracesToSkip = 20  # warm-up to avoid numerical problems for small evolution step

        CondAver = ConditionalAveragerAes(256, number_of_points)
        for i in range(tracesToSkip - 1):
            CondAver.addTrace(labels[i], samples[i])
        for i in range(tracesToSkip - 1, number_of_traces):
            CondAver.addTrace(labels[i], samples[i])

            if ((i + 1) % report_interval == 0) or ((i + 1) == number_of_traces):
                average_data, average_traces = CondAver.getSnapshot()
                correct_key = key[i][leakage_model["byte"]]
                corr_traces = self.cpa(average_data, average_traces, correct_key, leakage_model)

        return corr_traces.T
