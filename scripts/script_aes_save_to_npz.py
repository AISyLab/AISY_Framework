from neural_networks.neural_networks import *
from commons.sca_aisy_aes import Aisy

aisy = Aisy()
aisy.set_database_name("database_ascad.sqlite")
aisy.set_dataset("ascad-variable.h5")
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
aisy.set_epochs(20)
aisy.set_neural_network(mlp)
aisy.run(
    save_to_npz=["aes_attack"]
)
