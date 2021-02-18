def run(x_profiling, y_profiling, plaintexts_profiling,
        ciphertexts_profiling, key_profiling,
        x_validation, y_validation, plaintexts_validation,
        ciphertexts_validation, key_validation,
        x_attack, y_attack, plaintexts_attack,
        ciphertexts_attack, key_attack,
        param, aes_leakage_model,
        key_rank_executions, key_rank_report_interval, key_rank_attack_traces,
        model, *args):

    loss, acc = model.evaluate(x_validation, y_validation, verbose=0)
    return acc
