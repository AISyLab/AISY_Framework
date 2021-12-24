from aisy_sca.crypto.sca_aes_create_intermediates import *


def run(dataset, settings, model, *args):
    x_traces = dataset.x_validation[:500]
    plaintext = dataset.plaintext_validation[:500]
    if dataset.ciphertext_validation is not None:
        ciphertext = dataset.ciphertext_validation[:500]
    else:
        ciphertext = dataset.ciphertext_validation

    key_rank_attack_traces = 50
    key_rank_executions = 200

    nt = len(x_traces)

    # ---------------------------------------------------------------------------------------------------------#
    # compute labels for key hypothesis
    # ---------------------------------------------------------------------------------------------------------#
    labels_key_hypothesis = np.zeros((256, nt))
    for key_byte_hypothesis in range(0, 256):
        key_h = bytearray.fromhex(settings["key"])
        key_h[settings["leakage_model"]["byte"]] = key_byte_hypothesis
        labels_key_hypothesis[key_byte_hypothesis][:] = aes_intermediates(plaintext, ciphertext, key_h, settings["leakage_model"])

    good_key = settings["good_key"]

    # ---------------------------------------------------------------------------------------------------------#
    # predict output probabilities for shuffled test or validation set
    # ---------------------------------------------------------------------------------------------------------#
    output_probabilities = np.log(model.predict(x_traces) + 1e-36)

    probabilities_kg_all_traces = np.zeros((nt, 256))
    for index in range(nt):
        probabilities_kg_all_traces[index] = output_probabilities[index][
            np.asarray([int(leakage[index]) for leakage in labels_key_hypothesis[:]])  # array with 256 leakage values (1 per key guess)
        ]

    key_ranking_sum = 0
    for key_rank_execution in range(key_rank_executions):
        r = np.random.choice(range(nt), key_rank_attack_traces, replace=False)
        probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
        key_probabilities = np.sum(probabilities_kg_all_traces_shuffled[:key_rank_attack_traces], axis=0)
        key_probabilities_sorted = np.argsort(key_probabilities)[::-1]
        key_ranking_sum += list(key_probabilities_sorted).index(good_key) + 1

    guessing_entropy = key_ranking_sum / key_rank_executions

    print(f"Fast Guessing Entropy = {guessing_entropy}")

    return guessing_entropy
