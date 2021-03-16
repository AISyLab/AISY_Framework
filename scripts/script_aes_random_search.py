import aisy_sca
from app import *

aisy = aisy_sca.Aisy()
aisy.set_resources_root_folder(resources_root_folder)
aisy.set_database_root_folder(databases_root_folder)
aisy.set_datasets_root_folder(datasets_root_folder)
aisy.set_database_name("database_ascad.sqlite")
aisy.set_dataset(datasets_dict["ascad-variable.h5"])
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
aisy.set_epochs(10)

# for each hyper-parameter, specify the min, max and step or the possible options
random_search = {
    "neural_network": "cnn",
    "hyper_parameters_search": {
        'conv_layers': {"min": 1, "max": 2, "step": 1},
        'kernel_1': {"min": 2, "max": 8, "step": 1},
        'kernel_2': {"min": 2, "max": 8, "step": 1},
        'stride_1': {"min": 5, "max": 10, "step": 5},
        'stride_2': {"min": 5, "max": 10, "step": 5},
        'filters_1': {"min": 8, "max": 32, "step": 4},
        'filters_2': {"min": 8, "max": 32, "step": 4},
        'pooling_type_1': ["Average", "Max"],
        'pooling_type_2': ["Average", "Max"],
        'pooling_size_1': {"min": 1, "max": 1, "step": 1},
        'pooling_size_2': {"min": 1, "max": 1, "step": 1},
        'pooling_stride_1': {"min": 1, "max": 1, "step": 1},
        'pooling_stride_2': {"min": 1, "max": 1, "step": 1},
        'neurons': {"min": 100, "max": 1000, "step": 100},
        'layers': {"min": 2, "max": 3, "step": 1},
        'learning_rate': [0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001],
        'activation': ["relu", "selu", "elu", "tanh"],
        'epochs': {"min": 5, "max": 5, "step": 1},
        'batch_size': {"min": 100, "max": 1000, "step": 100},
        'optimizer': ["Adam", "RMSprop", "Adagrad", "Adadelta", "SGD"]
    },
    "metric": "guessing_entropy",
    "stop_condition": False,
    "stop_value": 1.0,
    "max_trials": 10,
    "train_after_search": True
}

aisy.run(
    random_search=random_search,
    key_rank_attack_traces=500
)
