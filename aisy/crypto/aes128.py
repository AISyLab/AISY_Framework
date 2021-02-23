import numpy as np

s_box = np.array([
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
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
])

inv_s_box = np.array([
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
])

shift_row_mask = np.array([0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11])
inv_shift_row_mask = np.array([0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3])


def sub_bytes(s):
    return s_box[s]


def inv_sub_bytes(s):
    return inv_s_box[s]


def shift_rows(s):
    s = np.array(s)
    return s[shift_row_mask]


def inv_shift_rows(s):
    s = np.array(s)
    return s[inv_shift_row_mask]


def add_round_key(s, k):
    return [s[i] ^ k[i] for i in range(len(s))]


xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)


def mix_single_column(a):
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    u = a[0]
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    a[3] ^= t ^ xtime(a[3] ^ u)
    return a


def mix_columns(s):
    for i in range(4):
        s[i * 4:i * 4 + 4] = mix_single_column(s[i * 4:i * 4 + 4])
    return s


def inv_mix_columns(s):
    for i in range(4):
        u = xtime(xtime(s[i * 4] ^ s[i * 4 + 2]))
        v = xtime(xtime(s[i * 4 + 1] ^ s[i * 4 + 3]))
        s[i * 4] ^= u
        s[i * 4 + 1] ^= v
        s[i * 4 + 2] ^= u
        s[i * 4 + 3] ^= v
    s = mix_columns(s)
    return s


r_con = (
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
    0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
)


def xor_bytes(a, b):
    return [a[i] ^ b[i] for i in range(len(a))]


def expand_key(master_key):
    iteration_count = 0
    for i in range(4, 44):
        word = list(master_key[len(master_key) - 4:])

        if i % 4 == 0:
            word.append(word.pop(0))
            word = [s_box[b] for b in word]
            word[0] ^= r_con[i // 4]

        word = xor_bytes(word, master_key[iteration_count * 4:iteration_count * 4 + 4])
        for w in word:
            master_key.append(w)

        iteration_count += 1

    return [master_key[16 * i: 16 * (i + 1)] for i in range(len(master_key) // 16)]


print_active = True


def print_state(state, method):
    if print_active:
        print("{}: {}".format(method.__name__, state))


def get_state_index_encryption(leakage_model):
    if leakage_model["target_round"] == 0 and leakage_model["target_state"] == 'AddRoundKey':
        return 0


def get_state_index_encryption_input():
    return {
        0: {'Input': 0, 'AddRoundKey': 1},
        1: {'Sbox': 2, 'ShiftRows': 3, 'MixColumns': 4, 'AddRoundKey': 5},
        2: {'Sbox': 6, 'ShiftRows': 7, 'MixColumns': 8}
    }


def get_state_index_encryption_output():
    return {
        10: {'Output': 0, 'AddRoundKey': 1, 'ShiftRows': 2, 'Sbox': 3},
        9: {'AddRoundKey': 4, 'MixColumns': 5, 'ShiftRows': 6, 'Sbox': 7}
    }


def get_state_index_decryption_input():
    return {
        1: {'Input': 0, 'AddRoundKey': 1, 'InvShiftRows': 2, 'InvSbox': 3},
        2: {'AddRoundKey': 4, 'InvMixColumns': 5, 'InvShiftRows': 6, 'InvSbox': 7}
    }


def get_state_index_decryption_output():
    return {
        10: {'Output': 0, 'AddRoundKey': 1},
        9: {'InvShiftRows': 2, 'InvSbox': 3, 'InvMixColumns': 4, 'AddRoundKey': 5},
        8: {'InvShiftRows': 6, 'InvSbox': 7, 'InvMixColumns': 8}
    }


def encrypt_block(plaintext, key):
    """
    :argument:
        plaintext (numpy array)
        key (numpy array)
    :return:
        ciphertext (numpy array)
    """

    round_keys = expand_key(key)
    state = add_round_key(plaintext, round_keys[0])

    for i in range(1, 10):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, round_keys[i])

    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, round_keys[-1])
    return state


def encrypt_block_leakage_model_from_input_fast(plaintexts, keys, leakage_model):

    """
    If AES leakage model is HW of S-Box output in the first encryption round, the fast method is called.
    :param plaintexts: 2D array containing the plaintexts for all traces
    :param keys: 2D array containing the keys for all traces
    :param leakage_model: leakage model dictionary
    :return: intermediates
    """

    plaintext = [row[leakage_model["byte"]] for row in plaintexts]
    key = [row[leakage_model["byte"]] for row in keys]
    state = [int(p) ^ int(k) for p, k in zip(np.asarray(plaintext[:]), np.asarray(key[:]))]

    return s_box[state]


def encrypt_block_leakage_model_from_input_fast_sr_ge(plaintexts, keys, leakage_model):

    """
    If AES leakage model is HW of S-Box output in the first encryption round, the fast method is called.
    This method is used to get intermediates for key candidates
    :param plaintexts: 2D array containing the plaintexts for all traces
    :param keys: 2D array containing the keys for all traces
    :param leakage_model: leakage model dictionary
    :return: intermediates
    """

    plaintext = [row[leakage_model["byte"]] for row in plaintexts]
    key = np.full(len(plaintext), keys[leakage_model["byte"]])
    state = [int(p) ^ int(k) for p, k in zip(np.asarray(plaintext[:]), key)]

    return s_box[state]


def encrypt_block_leakage_model_from_input(plaintext, key, leakage_model, state_index, state_index_first, state_index_second,
                                           round_keys=None):
    if round_keys is None:
        key_to_expand = key
        round_keys = expand_key(key_to_expand)
    states = []

    #  round 0
    states.append(plaintext)  # 0: Input
    state = add_round_key(plaintext, round_keys[0])
    states.append(state)  # 1: AddRoundKey
    #  round 1
    state = sub_bytes(state)
    states.append(state)  # 2: Sbox
    state = shift_rows(state)
    states.append(inv_shift_rows(state))  # 3: ShiftRows:
    state = mix_columns(state)
    states.append(inv_shift_rows(state))  # 4: MixColumns
    state = add_round_key(state, round_keys[1])
    states.append(inv_shift_rows(state))  # 5: AddRoundKey
    #  round 2
    state = sub_bytes(state)
    states.append(inv_shift_rows(state))  # 6: Sbox
    state = shift_rows(state)
    states.append(inv_shift_rows(inv_shift_rows(state)))  # 7: ShiftRows:
    state = mix_columns(state)
    states.append(inv_shift_rows(inv_shift_rows(state)))  # 8: MixColumns

    if leakage_model["leakage_model"] == "HD":
        return states[state_index_first][leakage_model["byte"]] ^ states[state_index_second][leakage_model["byte"]]
    else:
        return states[state_index][leakage_model["byte"]]


def encrypt_block_leakage_model_from_output(ciphertext, key, leakage_model, state_index, state_index_first, state_index_second,
                                            round_keys=None):
    if round_keys is None:
        key_to_expand = key
        round_keys = expand_key(key_to_expand)
    states = []

    # round 9
    states.append(ciphertext)  # 0: Output
    state = add_round_key(ciphertext, round_keys[10])
    states.append(state)  # 1: AddRoundKey
    state = inv_shift_rows(state)
    states.append(shift_rows(state))  # 2: ShiftRows
    state = inv_sub_bytes(state)
    states.append(shift_rows(state))  # 3: Sbox
    # round 8
    state = add_round_key(state, round_keys[9])
    states.append(shift_rows(state))  # 4: AddRoundKey
    state = inv_mix_columns(state)
    states.append(shift_rows(state))  # 5: MixColumns
    state = inv_shift_rows(state)
    states.append(shift_rows(shift_rows(state)))  # 6: ShiftRows
    state = inv_sub_bytes(state)
    states.append(shift_rows(shift_rows(state)))  # 7: Sbox

    if leakage_model["leakage_model"] == "HD":
        return states[state_index_first][leakage_model["byte"]] ^ states[state_index_second][leakage_model["byte"]]
    else:
        return states[state_index][leakage_model["byte"]]


def decrypt_block(ciphertext, key):
    """
    :argument:
        ciphertext (numpy array)
        key (numpy array)
    :return:
        plaintext (numpy array)
    """

    round_keys = expand_key(key)
    state = add_round_key(ciphertext, round_keys[10])
    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)

    for i in range(9, 0, -1):
        state = add_round_key(state, round_keys[i])
        state = inv_mix_columns(state)
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)

    return add_round_key(state, round_keys[0])


def decrypt_block_leakage_model_from_input(plaintext, key, leakage_model, state_index, state_index_first, state_index_second,
                                           round_keys=None):
    if round_keys is None:
        key_to_expand = key
        round_keys = expand_key(key_to_expand)
    states = []
    # round 1
    states.append(plaintext)  # 0: Input
    state = add_round_key(plaintext, round_keys[10])
    states.append(state)  # 1: AddRoundKey
    state = inv_shift_rows(state)
    states.append(state)  # 2: InvShiftRows
    state = inv_sub_bytes(state)
    states.append(shift_rows(state))  # 3: InvSbox
    # round 2
    state = add_round_key(state, round_keys[9])
    states.append(shift_rows(state))  # 4: AddRoundKey
    state = inv_mix_columns(state)
    states.append(shift_rows(state))  # 5: InvMixColumns
    state = inv_sub_bytes(state)
    states.append(shift_rows(state))  # 6: InvShiftRows
    state = inv_shift_rows(state)
    states.append(shift_rows(shift_rows(state)))  # 7: InvSbox

    if leakage_model["leakage_model"] == "HD":
        return states[state_index_first][leakage_model["byte"]] ^ states[state_index_second][leakage_model["byte"]]
    else:
        return states[state_index][leakage_model["byte"]]


def decrypt_block_leakage_model_from_output(ciphertext, key, leakage_model, state_index, state_index_first, state_index_second,
                                            round_keys=None):
    if round_keys is None:
        key_to_expand = key
        round_keys = expand_key(key_to_expand)
    states = []
    # round 10
    states.append(ciphertext)  # 0: Output
    state = add_round_key(ciphertext, round_keys[0])
    states.append(state)  # 1: AddRoundKey
    # round 9
    state = shift_rows(state)
    states.append(state)  # 2: InvShiftRows
    state = sub_bytes(state)
    states.append(inv_shift_rows(state))  # 3: InvSbox
    state = mix_columns(state)
    states.append(inv_shift_rows(state))  # 4: InvMixColumns
    state = add_round_key(state, round_keys[9])
    states.append(inv_shift_rows(state))  # 5: AddRoundKey
    # round 8
    state = shift_rows(state)
    states.append(inv_shift_rows(inv_shift_rows(state)))  # 6: InvShiftRows
    state = sub_bytes(state)
    states.append(inv_shift_rows(inv_shift_rows(state)))  # 7: InvSbox
    state = mix_columns(state)
    states.append(inv_shift_rows(inv_shift_rows(state)))  # 8: InvMixColumns

    if leakage_model["leakage_model"] == "HD":
        return states[state_index_first][leakage_model["byte"]] ^ states[state_index_second][leakage_model["byte"]]
    else:
        return states[state_index][leakage_model["byte"]]


def get_round_key(key, round):
    return expand_key(key)[round]


def run_test():
    # key as list of integets
    key = ([int(x) for x in bytearray.fromhex("2b7e151628aed2a6abf7158809cf4f3c")])
    plaintext = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    print(plaintext)

    ciphertext = encrypt_block(plaintext, key)
    print(ciphertext)

    key = ([int(x) for x in bytearray.fromhex("2b7e151628aed2a6abf7158809cf4f3c")])
    plaintext2 = decrypt_block(ciphertext, key)
    print(plaintext2)

# run_test()
