from aisy.sca_deep_learning_aes import AisyAes

aisy = AisyAes()
aisy.set_dataset("ascad-variable.h5")
aisy.set_database_name("database_ascad.sqlite")
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_number_of_profiling_traces(100000)
aisy.set_number_of_attack_traces(2000)
aisy.set_batch_size(400)
aisy.set_epochs(20)

# for each hyper-parameter, specify the options in the grid search
grid_search = {
    "neural_network": "mlp",
    "hyper_parameters_search": {
        'neurons': [50, 100],
        'layers': [2, 3],
        'learning_rate': [0.001, 0.0001],
        'activation': ["relu", "selu"]
    },
    "metric": "guessing_entropy",
    "stop_condition": False,
    "stop_value": 1.0,
    "train_after_search": True
}

aisy.run(
    key_rank_attack_traces=1000,
    grid_search=grid_search,
    ensemble=[10],
    probability_rank_plot=True
)
