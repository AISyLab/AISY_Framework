from custom.custom_models.neural_networks import *
from aisy.sca_deep_learning_aes import AisyAes
from custom.custom_data_augmentation.data_augmentation import *

aisy = AisyAes()
aisy.set_dataset("ascad-variable.h5")
aisy.set_database_name("database_ascad.sqlite")
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
aisy.set_epochs(20)
aisy.set_neural_network(mlp)
aisy.run(data_augmentation=[data_augmentation_gaussian_noise, 100])
