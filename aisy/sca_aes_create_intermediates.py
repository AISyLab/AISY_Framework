import numpy as np
from aisy.crypto.aes128 import *


def state_aes_leakage_model_dict():
    return {
        "input": {
            "Encryption": get_state_index_encryption_input,
            "Decryption": get_state_index_decryption_input
        },
        "output": {
            "Encryption": get_state_index_encryption_output,
            "Decryption": get_state_index_decryption_output,
        }
    }


def aes_leakage_model_dict():
    return {
        "input": {
            "Encryption": encrypt_block_leakage_model_from_input,
            "Decryption": decrypt_block_leakage_model_from_input
        },
        "output": {
            "Encryption": encrypt_block_leakage_model_from_output,
            "Decryption": decrypt_block_leakage_model_from_output,
        }
    }


def aes_intermediates(plaintexts, ciphertexts, keys, leakage_model):
    if leakage_model["attack_direction"] == "input" and leakage_model["direction"] == "Encryption":
        intermediate_values = encrypt_block_leakage_model_from_input_fast(plaintexts, keys, leakage_model)
    else:
        intermediate_values = []
        state_aes_leakage_model_method = state_aes_leakage_model_dict()[leakage_model["attack_direction"]][leakage_model["direction"]]
        if leakage_model["leakage_model"] == "HD":
            state_index = 0
            state_index_first = state_aes_leakage_model_method()[leakage_model['round_first']][leakage_model['target_state_first']]
            state_index_second = state_aes_leakage_model_method()[leakage_model['round_second']][leakage_model['target_state_second']]
        else:
            state_index = state_aes_leakage_model_method()[leakage_model['round']][leakage_model['target_state']]
            state_index_first = 0
            state_index_second = 0
        aes_leakage_model_method = aes_leakage_model_dict()[leakage_model["attack_direction"]][leakage_model["direction"]]
        if leakage_model["attack_direction"] == "input":
            keys = np.array(keys, dtype=np.uint8)
            plaintexts = np.array(plaintexts, dtype=np.uint8)
            for plaintext, key in zip(plaintexts, keys):
                intermediate_values.append(aes_leakage_model_method(list(plaintext), list(key), leakage_model, state_index,
                                                                    state_index_first, state_index_second))
        else:
            keys = np.array(keys, dtype=np.uint8)
            ciphertexts = np.array(ciphertexts, dtype=np.uint8)
            for ciphertext, key in zip(ciphertexts, keys):
                intermediate_values.append(aes_leakage_model_method(list(ciphertext), list(key), leakage_model, state_index,
                                                                    state_index_first, state_index_second))

    if leakage_model["leakage_model"] == "HW" or leakage_model["leakage_model"] == "HD":
        return [bin(iv).count("1") for iv in intermediate_values]
    elif leakage_model["leakage_model"] == "bit":
        return [int(bin(iv >> leakage_model["bit"])[len(bin(iv >> leakage_model["bit"])) - 1]) for iv in intermediate_values]
    else:
        return intermediate_values


def aes_intermediates_sr_ge(plaintexts, ciphertexts, key_candidate, leakage_model):
    if leakage_model["attack_direction"] == "input" and leakage_model["direction"] == "Encryption":
        intermediate_values = encrypt_block_leakage_model_from_input_fast_sr_ge(plaintexts, key_candidate, leakage_model)
    else:
        intermediate_values = []
        state_aes_leakage_model_method = state_aes_leakage_model_dict()[leakage_model["attack_direction"]][leakage_model["direction"]]
        if leakage_model["leakage_model"] == "HD":
            state_index = 0
            state_index_first = state_aes_leakage_model_method()[leakage_model['round_first']][leakage_model['target_state_first']]
            state_index_second = state_aes_leakage_model_method()[leakage_model['round_second']][leakage_model['target_state_second']]
        else:
            state_index = state_aes_leakage_model_method()[leakage_model['round']][leakage_model['target_state']]
            state_index_first = 0
            state_index_second = 0
        aes_leakage_model_method = aes_leakage_model_dict()[leakage_model["attack_direction"]][leakage_model["direction"]]
        if leakage_model["attack_direction"] == "input":
            key = np.array(key_candidate, dtype=np.uint8)
            round_keys = expand_key(list(key))
            plaintexts = np.array(plaintexts, dtype=np.uint8)
            for plaintext in plaintexts:
                intermediate_values.append(aes_leakage_model_method(list(plaintext), list(key), leakage_model, state_index,
                                                                    state_index_first, state_index_second, round_keys))
        else:
            key = np.array(key_candidate, dtype=np.uint8)
            round_keys = expand_key(list(key))
            ciphertexts = np.array(ciphertexts, dtype=np.uint8)
            for ciphertext in ciphertexts:
                intermediate_values.append(aes_leakage_model_method(list(ciphertext), list(key), leakage_model, state_index,
                                                                    state_index_first, state_index_second, round_keys))

    if leakage_model["leakage_model"] == "HW" or leakage_model["leakage_model"] == "HD":
        return [bin(iv).count("1") for iv in intermediate_values]
    elif leakage_model["leakage_model"] == "bit":
        return [int(bin(iv >> leakage_model["bit"])[len(bin(iv >> leakage_model["bit"])) - 1]) for iv in intermediate_values]
    else:
        return intermediate_values
