import aisy_sca
from app import *
from custom.custom_models.neural_networks import *
from aisy_sca.datasets.Dataset import Dataset
from tensorflow.keras.utils import to_categorical

""" set profiling, validation and attack traces"""
x_profiling = np.random.rand(10000, 100)
x_attack = np.random.rand(1000, 100)
x_validation = np.random.rand(1000, 100)

""" set profiling, validation and attack labels"""
y_profiling = to_categorical(np.random.randint(0, 256, 10000), num_classes=256)
y_attack = to_categorical(np.random.randint(0, 256, 1000), num_classes=256)
y_validation = to_categorical(np.random.randint(0, 256, 1000), num_classes=256)

""" create list of key guesses for attack and validation sets """
labels_key_guess_validation_set = np.random.randint(0, 256, (256, 1000))
labels_key_guess_attack_set = np.random.randint(0, 256, (256, 1000))

""" create dataset object """
new_dataset = Dataset(x_profiling, y_profiling, x_attack, y_attack, x_validation, y_validation)

new_dataset_dict = {
    "filename": "new_dataset.h5",
    "key": "4DFBE0F27221FE10A78D4ADC8E490469",
    "first_sample": 0,
    "number_of_samples": 100,
    "number_of_profiling_traces": 10000,
    "number_of_attack_traces": 1000
}

aisy = aisy_sca.Aisy()
aisy.set_resources_root_folder(resources_root_folder)
aisy.set_database_root_folder(databases_root_folder)
aisy.set_datasets_root_folder(datasets_root_folder)
aisy.set_database_name("database_ascad_variable.sqlite")

""" User must set dataset object and value of good key for GE and SR """
aisy.set_classes(256)
aisy.set_good_key(224)
aisy.set_dataset(new_dataset_dict, dataset=new_dataset)
aisy.set_labels_key_guesses_attack_set(labels_key_guess_attack_set)
aisy.set_labels_key_guesses_validation_set(labels_key_guess_validation_set)

aisy.set_batch_size(400)
aisy.set_epochs(10)
aisy.add_neural_network(mlp)
aisy.run()
