import aisy_sca
from app import *
from custom.custom_models.neural_networks import *

aisy = aisy_sca.Aisy()
aisy.set_resources_root_folder(resources_root_folder)
aisy.set_database_root_folder(databases_root_folder)
aisy.set_datasets_root_folder(datasets_root_folder)
aisy.set_database_name("database_ascad.sqlite")
aisy.set_dataset(datasets_dict["ASCAD.h5"])
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
aisy.set_epochs(50)
aisy.add_neural_network(cnn_architecture, name="model_0")
aisy.add_neural_network(mlp, name="model_3")
aisy.add_neural_network(noConv1_ascad_desync_0, name="model_1")
aisy.add_neural_network(methodology_cnn_ascad, name="model_2")
aisy.add_neural_network(cnn, name="model_4")
aisy.run()
