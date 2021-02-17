from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import *
from sklearn.utils import shuffle
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import signal
import scipy


class CustomCallback1(Callback):
    def __init__(self,
                 x_profiling, y_profiling, plaintexts_profiling,
                 x_validation, y_validation, plaintexts_validation,
                 x_attack, y_attack, plaintexts_attack,
                 param, aes_leakage_model,
                 key_rank_executions, key_rank_report_interval, key_rank_attack_traces,
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
                 x_validation, y_validation, plaintexts_validation,
                 x_attack, y_attack, plaintexts_attack,
                 param, aes_leakage_model,
                 key_rank_executions, key_rank_report_interval, key_rank_attack_traces,
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


class GetActivations(Callback):
    def __init__(self,
                 x_profiling, y_profiling, plaintexts_profiling,
                 x_validation, y_validation, plaintexts_validation,
                 x_attack, y_attack, plaintexts_attack,
                 param, aes_leakage_model,
                 key_rank_executions, key_rank_report_interval, key_rank_attack_traces):
        self.x = x_attack
        self.activations_dict_epochs = []
        self.activations_dict = {}

    def on_epoch_end(self, epoch, logs=None):
        layer_name = None
        outputs = [layer.output for layer in self.model.layers if layer.name == layer_name or layer_name is None]

        self.activations_dict = {}
        # loop over each layer and get output activations
        for index, output in enumerate(outputs):
            intermediate_model = Model(inputs=self.model.input, outputs=output)
            layer_activations = intermediate_model.predict(self.x)
            print('Layer {} with activation shape {}'.format(index, np.shape(layer_activations)))
            self.activations_dict['activations_{}'.format(index + 1)] = layer_activations

        self.activations_dict_epochs.append(self.activations_dict)

        act_shape = np.shape(self.activations_dict['activations_1'])
        for i in range(act_shape[2]):
            plt.subplot(4, 4, i + 1)
            # plt.subplot(2, 1, 1)
            plt.magnitude_spectrum(self.activations_dict['activations_1'][0, :, i], Fs=2e9)
        if epoch == 49:
            plt.show()

    def on_train_end(self, logs=None):
        pass

    def get_activations(self):
        return self.activations_dict

    def get_activations_epochs(self):
        return self.activations_dict_epochs


class GetWeights(Callback):
    def __init__(self,
                 x_profiling, y_profiling, plaintexts_profiling,
                 x_validation, y_validation, plaintexts_validation,
                 x_attack, y_attack, plaintexts_attack,
                 param, aes_leakage_model,
                 key_rank_executions, key_rank_report_interval, key_rank_attack_traces):
        self.weight_dict_epochs = []
        self.weight_dict = {}
        # self.h = []
        # self.w = None

    def on_epoch_end(self, epoch, logs=None):
        self.weight_dict = {}

        # loop over each layer and get weights and biases
        for index, layer in enumerate(self.model.layers):
            if len(layer.get_weights()) > 0:
                w = layer.get_weights()[0]
                b = layer.get_weights()[1]
                print('Layer {} has weights of shape {} and biases of shape {}'.format(index, np.shape(w), np.shape(b)))

                self.weight_dict['w_{}'.format(index + 1)] = w
                self.weight_dict['b_{}'.format(index + 1)] = b

        self.weight_dict_epochs.append(self.weight_dict)

        w_shape = np.shape(self.weight_dict['w_1'])
        for i in range(w_shape[2]):
            w, h = signal.freqz(self.weight_dict['w_1'][:, 0, i])
            # self.h.append(h)
            # self.w = w
            plt.subplot(4, 4, 9 + i)
            # plt.subplot(2, 1, 2)
            plt.plot(w / 3.1428, 20 * np.log10(np.abs(h)))
        if epoch == 49:
            # plt.plot(self.w / 3.1428, 20 * np.log10(np.abs(np.mean(self.h, axis=0))))
            plt.show()

    def on_train_end(self, logs=None):
        pass

    def get_weights(self):
        return self.weight_dict

    def get_weights_epochs(self):
        return self.weight_dict_epochs


class GetLinearSumG(Callback):
    def __init__(self,
                 x_profiling, y_profiling, plaintexts_profiling,
                 x_validation, y_validation, plaintexts_validation,
                 x_attack, y_attack, plaintexts_attack,
                 param, aes_leakage_model,
                 key_rank_executions, key_rank_report_interval, key_rank_attack_traces):
        self.x_profiling = x_profiling
        self.gxn_epochs = []

    def on_epoch_end(self, epoch, logs=None):

        print("\n")

        layer_name = None
        outputs = [layer.output for layer in self.model.layers if layer.name == layer_name or layer_name is None]

        self.gxn_dict = {}
        for index, layer in enumerate(self.model.layers):

            if len(layer.get_weights()) > 0:
                w = layer.get_weights()[0]
                b = layer.get_weights()[1]
                print('Layer {} has weights of shape {} and biases of shape {}'.format(index, np.shape(w), np.shape(b)))

                if index > 0:
                    intermediate_model = Model(inputs=self.model.input, outputs=outputs[index - 1])
                    layer_activations = intermediate_model.predict(self.x_profiling)
                    print('Layer {} with activation shape {}'.format(index, np.shape(layer_activations)))
                else:
                    layer_activations = self.x_profiling
                    print('Layer {} with activation shape {}'.format(index, np.shape(layer_activations)))

                g = [layer_activations.dot(w[:, neuron_index]) + b[neuron_index] for neuron_index in range(np.shape(w)[1])]
                g = np.transpose(g)
                f = np.maximum(g, 0)  # this is not needed, as we can obtain from 'layer_activations' output
                print('Linear Sum G{}n with shape {}'.format(index, np.shape(g)))
                print('Activation F{}n with shape {}'.format(index, np.shape(f)))

                self.gxn_dict['g{}n'.format(index + 1)] = g

        self.gxn_epochs.append(self.gxn_dict)

    def get_gxn_epochs(self):
        return self.gxn_epochs


###############################################################################
# Calculate the entropy of given distribution 'pdf'
def cal_entropy(pdf):
    # Guarantee there is no zero proablity
    pdf1 = np.transpose(pdf + np.spacing(1))
    entropy = np.sum(np.multiply(pdf1, np.log2(1 / pdf1)), axis=0)
    return entropy


###############################################################################
# Calculate the entropy of given distribution 'pdf'
def cal_entropy1(pdf):
    pdf = np.transpose(pdf)
    # print(pdf.shape)
    n_samples = pdf.shape[1]
    if n_samples == 1:
        pdf1 = pdf + np.spacing(1)
        pdf1 = pdf1 / np.sum(pdf1)
        entropy = np.sum(np.multiply(pdf1, np.log2(1 / pdf1)))
        return entropy
    else:
        entropy = np.zeros((n_samples, 1))
        for i in range(n_samples):
            pdf1 = pdf[:, i] + np.spacing(1)
            pdf1 = pdf1 / np.sum(pdf1)
            entropy1 = np.sum(np.multiply(pdf1, np.log2(1 / pdf1)))
            entropy[i] = entropy1
        return entropy


###############################################################################
# Derive the Gibbs distribution based the activations of a hidden layer
def Gibbs_pdf(energy):
    energy = np.float64(energy)
    exp_energy = np.exp(energy + np.spacing(1))
    partition = np.sum(exp_energy, axis=1)
    gibbs = exp_energy / (partition[:, None])
    gibbs = np.float32(gibbs)
    return gibbs


###############################################################################
# Calculate the mutual information between F and X
def cal_MIFX(pdf_all):
    entropy_all = cal_entropy(pdf_all)
    conditional_entropy_all = np.mean(entropy_all)
    pdf_F = np.mean(pdf_all, axis=0)
    pdf_F = (pdf_F / np.sum(pdf_F)).reshape(1, -1)
    entropy_F = cal_entropy(pdf_F)
    MIFX = entropy_F - conditional_entropy_all
    return MIFX


###############################################################################
# Calculate the mutual information between F and Y
def cal_MIFY(pdf_all, y_true_batch):
    label_mark = np.argmax(y_true_batch, axis=1)

    pdf_F = np.mean(pdf_all, axis=0)
    pdf_F = (pdf_F / np.sum(pdf_F)).reshape(1, -1)
    entropy_F = cal_entropy(pdf_F)

    H_FY_sum = 0
    for i in range(9):
        labeli = np.argwhere(label_mark == i)
        pdf_fyi = np.squeeze(pdf_all[labeli, :])
        pdf_FYi = np.mean(pdf_fyi, axis=0)
        pdf_FYi = (pdf_FYi / np.sum(pdf_FYi)).reshape(1, -1)
        entropy_FYi = cal_entropy(pdf_FYi)
        H_FY_sum += entropy_FYi
    MIFY = entropy_F - H_FY_sum / 10
    return MIFY


class MutualInformation(Callback):
    def __init__(self,
                 x_profiling, y_profiling, plaintexts_profiling,
                 x_validation, y_validation, plaintexts_validation,
                 x_attack, y_attack, plaintexts_attack,
                 param, aes_leakage_model,
                 key_rank_executions, key_rank_report_interval, key_rank_attack_traces):
        self.x_profiling = x_profiling
        self.x_labels = y_profiling
        self.MIFX_vector_epochs = []
        self.MIFY_vector_epochs = []
        self.MIFXO_vector_epochs = []

    def on_epoch_end(self, epoch, logs=None):
        layer_name = None
        outputs = [layer.output for layer in self.model.layers if layer.name == layer_name or layer_name is None]

        MIFX_vector = []
        MIFY_vector = []
        MIFXO_vector = []
        # loop over each layer and get output activations
        print("\n")
        for index, output in enumerate(outputs):
            intermediate_model = Model(inputs=self.model.input, outputs=output)
            layer_activations = intermediate_model.predict(self.x_profiling)

            # Derive the distribution P(F1)
            Gibbs_layer = Gibbs_pdf(layer_activations)

            MIFX_vector.append(cal_MIFX(Gibbs_layer))
            MIFY_vector.append(cal_MIFY(Gibbs_layer, self.x_labels))
            MIFXO_vector.append(MIFX_vector[index] - MIFY_vector[index])

            print('MI(F%d,X),  MI(F%d,Y),  and MI(F%d,XO) are (%.2f, %.2f, %.2f)' % (index + 1, index + 1, index + 1,
                                                                                     MIFX_vector[index],
                                                                                     MIFY_vector[index],
                                                                                     MIFXO_vector[index]))

        self.MIFX_vector_epochs.append(MIFX_vector)
        self.MIFY_vector_epochs.append(MIFY_vector)
        self.MIFXO_vector_epochs.append(MIFXO_vector)

    def get_MIFX_vector(self):
        return self.MIFX_vector_epochs

    def get_MIFY_vector(self):
        return self.MIFY_vector_epochs

    def get_MIFXO_vector(self):
        return self.MIFXO_vector_epochs


AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])


# labelize for key guesess for guessing entropy and success rate
def aes_labelize_ge_sr(plt_attack, byte, key, leakage):
    pt_ct = [row[byte] for row in plt_attack]

    key_byte = np.full(len(pt_ct), key[byte])
    state = [int(x) ^ int(k) for x, k in zip(np.asarray(pt_ct[:]), key_byte)]

    intermediate_values = AES_Sbox[state]

    if leakage == "HW":
        return [bin(iv).count("1") for iv in intermediate_values]
    else:
        return intermediate_values


# guessing entropy and success rate
def compute_ge(runs, model, key, correct_key, leakage_model, byte, x_attack, plt_attack, key_rank_report_interval, key_rank_attack_traces):
    nt = len(x_attack)
    nt_interval = int(key_rank_attack_traces / key_rank_report_interval)
    key_ranking_sum = np.zeros(nt_interval)

    # ---------------------------------------------------------------------------------------------------------#
    # compute labels for key hypothesis
    # ---------------------------------------------------------------------------------------------------------#
    labels_key_hypothesis = np.zeros((256, nt))
    for key_byte_hypothesis in range(0, 256):
        key_h = bytearray.fromhex(key)
        key_h[byte] = key_byte_hypothesis
        labels_key_hypothesis[key_byte_hypothesis][:] = aes_labelize_ge_sr(plt_attack, byte, key_h, leakage_model)

    # ---------------------------------------------------------------------------------------------------------#
    # predict output probabilities for shuffled test or validation set
    # ---------------------------------------------------------------------------------------------------------#
    output_probabilities = model.predict(x_attack)

    probabilities_kg_all_traces = np.zeros((nt, 256))
    for index in range(nt):
        probabilities_kg_all_traces[index] = output_probabilities[index][
            np.asarray([int(leakage[index]) for leakage in labels_key_hypothesis[:]])
        ]

    # ---------------------------------------------------------------------------------------------------------#
    # run key rank "runs" times and average results.
    # ---------------------------------------------------------------------------------------------------------#
    for run in range(runs):

        probabilities_kg_all_traces_shuffled = shuffle(probabilities_kg_all_traces,
                                                       random_state=random.randint(0, 100000))

        key_probabilities = np.zeros(256)

        kr_count = 0
        for index in range(key_rank_attack_traces):

            key_probabilities += np.log(probabilities_kg_all_traces_shuffled[index] + 1e-36)
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]

            if (index + 1) % key_rank_report_interval == 0:
                key_ranking_good_key = list(key_probabilities_sorted).index(correct_key) + 1
                key_ranking_sum[kr_count] += key_ranking_good_key
                kr_count += 1

        print("KR run: {} | final GE for correct key ({}): {})".format(run, correct_key, key_ranking_sum[nt_interval - 1] / (run + 1)))

    guessing_entropy = key_ranking_sum / runs

    return guessing_entropy


class GuessingEntropy(Callback):
    def __init__(self,
                 x_profiling, y_profiling, plaintexts_profiling,
                 x_validation, y_validation, plaintexts_validation,
                 x_attack, y_attack, plaintexts_attack,
                 param, aes_leakage_model,
                 key_rank_executions, key_rank_report_interval, key_rank_attack_traces):
        self.key_rank_runs = key_rank_executions
        self.key = param["key"]
        self.correct_key = param["good_key"]
        self.l_model = aes_leakage_model["leakage_model"]
        self.target_byte = aes_leakage_model["byte"]
        self.X_attack = x_attack
        self.attack_data = plaintexts_attack
        self.key_rank_report_interval = key_rank_report_interval
        self.key_rank_number_of_traces = key_rank_attack_traces
        self.ge_epochs = []

    def on_epoch_end(self, epoch, logs=None):
        ge = compute_ge(self.key_rank_runs, self.model, self.key, self.correct_key, self.l_model, self.target_byte, self.X_attack,
                        self.attack_data, self.key_rank_report_interval, self.key_rank_number_of_traces)
        self.ge_epochs.append(ge[len(ge) - 1])

    def get_ge_epochs(self):
        return self.ge_epochs
