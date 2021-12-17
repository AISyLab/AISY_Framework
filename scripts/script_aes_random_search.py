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

"""
convolution filter options:
- min, max, step
- "equal_from_previous_convolution"
- "double_from_previous_convolution"
- "half_from_previous_convolution"
- min="previous_convolution_filters" (with max and step)
- max="previous_convolution_filters" (with min and step)

convolution kernel options:
- min, max, step
- "equal_from_previous_convolution"
- "double_from_previous_convolution"
- "half_from_previous_convolution"
- min="previous_convolution_kernel" (with max and step)
- max="previous_convolution_kernel" (with min and step)

convolution stride options:
- min, max, step
- "equal_from_previous_convolution"
- "double_from_previous_convolution"
- "half_from_previous_convolution"
- min="previous_convolution_stride" (with max and step)
- max="previous_convolution_stride" (with min and step)

pooling size options:
- min, max, step
- "equal_from_previous_pooling"
- "double_from_previous_pooling"
- "half_from_previous_pooling"
- min="previous_pooling_size" (with max and step)
- max="previous_pooling_size" (with min and step)

pooling stride options:
- min, max, step
- "equal_from_previous_pooling"
- "double_from_previous_pooling"
- "half_from_previous_pooling"
- min="previous_pooling_stride" (with max and step)
- max="previous_pooling_stride" (with min and step)

pooling type options: 
- "Average", "Max"
- "equal_from_previous_pooling"

"""

# for each hyper-parameter, specify the min, max and step or the possible options
random_search = {
    "neural_network": "cnn",
    "hyper_parameters_search": {
        'conv_layers': {"min": 1, "max": 3, "step": 1},
        'kernel_1': {"min": 4, "max": 20, "step": 1},
        'kernel_2': {"min": 4, "max": 20, "step": 1},
        'kernel_3': {"min": 4, "max": 20, "step": 1},
        'stride_1': {"min": 1, "max": 4, "step": 1},
        'stride_2': {"min": 1, "max": 4, "step": 1},
        'stride_3': {"min": 1, "max": 4, "step": 1},
        'filters_1': {"min": 8, "max": 32, "step": 4},
        'filters_2': "double_from_previous_convolution",
        'filters_3': "double_from_previous_convolution",
        'pooling_type_1': ["Average", "Max"],
        'pooling_type_2': "equal_from_previous_pooling",
        'pooling_type_3': "equal_from_previous_pooling",
        'pooling_size_1': {"min": 1, "max": 1, "step": 1},
        'pooling_size_2': "equal_from_previous_pooling",
        'pooling_size_3': "equal_from_previous_pooling",
        'pooling_stride_1': {"min": 2, "max": 2, "step": 1},
        'pooling_stride_2': {"min": 2, "max": 2, "step": 1},
        'pooling_stride_3': {"min": 2, "max": 2, "step": 1},
        'neurons': {"min": 100, "max": 400, "step": 100},
        'layers': {"min": 2, "max": 3, "step": 1},
        'learning_rate': [0.005, 0.001, 0.0005, 0.0001],
        'activation': ["relu", "selu"],
        # 'epochs': {"min": 5, "max": 5, "step": 1},
        'batch_size': {"min": 100, "max": 1000, "step": 100},
        'optimizer': ["Adam", "RMSprop"]
    },
    "structure": {
        "use_pooling_after_convolution": True,  # only for CNNs
        "use_pooling_before_first_convolution": False,
        "use_pooling_before_first_dense": False,  # only for MLPs
        "use_batch_norm_after_pooling": True,
        "use_batch_norm_before_pooling": False,
        "use_batch_norm_after_convolution": False,
        "use_dropout_after_dense_layer": False,
        "use_dropout_before_dense_layer": False,
    },
    "metric": "guessing_entropy",
    "stop_condition": False,
    "stop_value": 1.0,
    "max_trials": 50,
    "train_after_search": True
}

aisy.run(
    random_search=random_search,
    key_rank_attack_traces=500
)
