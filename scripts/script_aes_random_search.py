from aisy.sca_deep_learning_aes import AisyAes

aisy = AisyAes()
# aisy.set_datasets_root_folder("D:/traces/")
aisy.set_dataset("ascad-variable.h5")
aisy.set_database_name("database_ascad.sqlite")
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_number_of_profiling_traces(50000)
aisy.set_number_of_attack_traces(1000)
aisy.set_batch_size(400)
aisy.set_epochs(20)

# for each hyper-parameter, specify the min, max and step or the possible options
random_search = {
    "neural_network": "cnn",
    "hyper_parameters_search": {
        'conv_layers': {"min": 1, "max": 2, "step": 1},
        'kernel_1': {"min": 2, "max": 8, "step": 1},
        'kernel_2': {"min": 2, "max": 8, "step": 1},
        'stride_1': {"min": 1, "max": 2, "step": 1},
        'stride_2': {"min": 1, "max": 2, "step": 1},
        'filters_1': {"min": 8, "max": 8, "step": 8},
        'filters_2': {"min": 16, "max": 16, "step": 16},
        'pooling_type_1': ["Average", "Max"],
        'pooling_type_2': ["Average", "Max"],
        'pooling_size_1': {"min": 2, "max": 2, "step": 1},
        'pooling_size_2': {"min": 2, "max": 2, "step": 1},
        'pooling_stride_1': {"min": 2, "max": 2, "step": 1},
        'pooling_stride_2': {"min": 2, "max": 2, "step": 1},
        'neurons': {"min": 10, "max": 1000, "step": 10},
        'layers': {"min": 1, "max": 10, "step": 1},
        'learning_rate': [0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005],
        'activation': ["relu", "selu", "elu", "tanh"],
        'epochs': {"min": 5, "max": 15, "step": 1},
        'mini_batch': {"min": 100, "max": 2000, "step": 100},
    },
    "metric": "guessing_entropy",
    "stop_condition": True,
    "stop_value": 1.0,
    "max_trials": 8,
    "train_after_search": True
}

aisy.run(
    random_search=random_search
)
