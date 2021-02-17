import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.python.keras import backend as K
import numpy as np
import scipy.stats as ss
import random

classes = 9
correct_key = 34

# # Compute the evolution of rank
# def rank_compute(prediction, att_plt, byte, output_rank):
#     hw = [bin(x).count("1") for x in range(256)]
#     (nb_traces, nb_hyp) = prediction.shape
#
#     key_log_prob = np.zeros(256)
#     prediction = np.log(prediction + 1e-40)
#     rank_evol = np.full(nb_traces, 255)
#
#     for i in range(nb_traces):
#         for k in range(256):
#             # Computes the hypothesis values
#             if leakage == 'Sbox':
#                 key_log_prob[k] += prediction[i, AES_Sbox[k ^ int(att_plt[i, byte])]]
#             else:
#                 key_log_prob[k] += prediction[i, hw[AES_Sbox[k ^ int(att_plt[i, byte])]]]
#         rank_evol[i] = rk_key(key_log_prob, correct_key)
#
#     if output_rank:
#         return rank_evol
#     else:
#         return key_log_prob


# def perform_attacks(nb_traces, predictions, plt_attack, nb_attacks=1, byte=0, shuffle=True, output_rank=False):
#     (nb_total, nb_hyp) = predictions.shape
#     all_rk_evol = np.zeros((nb_attacks, nb_traces))
#
#     for i in range(nb_attacks):
#         if shuffle:
#             l = list(zip(predictions, plt_attack))
#             random.shuffle(l)
#             sp, splt = list(zip(*l))
#             sp = np.array(sp)
#             splt = np.array(splt)
#             att_pred = sp[:nb_traces]
#             att_plt = splt[:nb_traces]
#
#         else:
#             att_pred = predictions[:nb_traces]
#             att_plt = plt_attack[:nb_traces]
#
#         key_log_prob = rank_compute(att_pred, att_plt, byte, output_rank)
#         if output_rank:
#             all_rk_evol[i] = key_log_prob
#
#     if output_rank:
#         return np.mean(all_rk_evol, axis=0)
#     else:
#         return np.float32(key_log_prob)

# calculate key prob for all keys
def calculate_key_prob(y_true, y_pred):
    # if plt_attack[0][32] == 46:  # check if data is from validation set, then compute GE
    #     GE = perform_attacks(nb_traces_attacks, y_pred, plt_attack, nb_attacks, byte=0)
    # else:  # otherwise, return zeros
    #     GE = np.float32(np.zeros(256))
    # return GE
    return np.float32(np.zeros(256))

@tf.function
def tf_calculate_key_prob(y_true, y_pred):
    _ret = tf.numpy_function(calculate_key_prob, [y_true, y_pred], tf.float32)
    return _ret

def calculate_rank(y_pred):
    pred_rank = ss.rankdata(y_pred, axis=1) - 1
    return pred_rank / 255


def rk_key(rank_array, key):
    key_val = rank_array[key]
    return np.float32(np.where(np.sort(rank_array)[::-1] == key_val)[0][0])


class key_rank_Metric(tk.metrics.Metric):
    def __init__(self, name='key_rank', **kwargs):
        super(key_rank_Metric, self).__init__(name=name, **kwargs)
        self.acc_sum = self.add_weight(name='acc_sum', shape=(256), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.acc_sum.assign_add(tf_calculate_key_prob(y_true, y_pred))

    def result(self):
        return tf.numpy_function(rk_key, [self.acc_sum, correct_key], tf.float32)

    def reset_states(self):
        self.acc_sum.assign(K.zeros(256))