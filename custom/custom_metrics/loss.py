def run(x_profiling, y_profiling, plaintexts_profiling,
        x_validation, y_validation, plaintexts_validation,
        x_attack, y_attack, plaintexts_attack,
        param, aes_leakage_model,
        key_rank_executions, key_rank_report_interval, key_rank_attack_traces,
        model, *args):

    loss, _ = model.evaluate(x_validation, y_validation, verbose=0)
    return loss