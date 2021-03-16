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
aisy.set_epochs(20)

# for each hyper-parameter, specify the options in the grid search
grid_search = {
    "neural_network": "cnn",
    "hyper_parameters_search": {
        'conv_layers': [1, 2],
        'kernel_1': [4, 8],
        'kernel_2': [2, 4],
        'stride_1': [1],
        'stride_2': [1],
        'filters_1': [8, 16],
        'filters_2': [8, 16],
        'pooling_type_1': ["Average", "Max"],
        'pooling_type_2': ["Average", "Max"],
        'pooling_size_1': [1, 2],
        'pooling_size_2': [1, 2],
        'pooling_stride_1': [1, 2],
        'pooling_stride_2': [1, 2],
        'neurons': [100, 200],
        'layers': [3, 4],
        'learning_rate': [0.001],
        'activation': ["selu", "elu"],
        'optimizer': ["Adam", "SGD"]
    },
    "metric": "guessing_entropy",
    "stop_condition": False,
    "stop_value": 1.0,
    "train_after_search": True
}

aisy.run(
    grid_search=grid_search,
    key_rank_attack_traces=500
)
