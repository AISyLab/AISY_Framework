from neural_networks.neural_networks import *
from commons.sca_aisy_aes import Aisy
from custom.custom_data_augmentation.data_augmentation import *

aisy = Aisy()
aisy.set_dataset("ascad-variable.h5")
aisy.set_database_name("database_ascad.sqlite")
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
aisy.set_epochs(20)
aisy.set_neural_network(mlp)
aisy.run(data_augmentation=[data_augmentation_gaussian_noise, 100])
