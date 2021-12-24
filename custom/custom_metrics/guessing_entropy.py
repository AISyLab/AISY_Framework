from aisy_sca.crypto.sca_aes_create_intermediates import *


def run(dataset, settings, model, *args):
    nt = len(dataset.x_validation)

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

    key_ranking_sum = 0
    for key_rank_execution in range(settings["key_rank_executions"]):
        r = np.random.choice(range(nt), settings["key_rank_attack_traces"], replace=False)
        probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
        key_probabilities = np.sum(probabilities_kg_all_traces_shuffled[:settings["key_rank_attack_traces"]], axis=0)
        key_probabilities_sorted = np.argsort(key_probabilities)[::-1]
        key_ranking_sum += list(key_probabilities_sorted).index(good_key) + 1

    guessing_entropy = key_ranking_sum / settings["key_rank_executions"]

    print(f"GE = {guessing_entropy}")

    return guessing_entropy
