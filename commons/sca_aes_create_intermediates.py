import numpy as np
from commons.crypto.aes128 import *

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
