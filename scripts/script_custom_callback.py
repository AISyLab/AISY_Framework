from custom.custom_models.neural_networks import *
from commons.sca_aisy_aes import Aisy
from custom.custom_callbacks.callbacks import *

aisy = Aisy()
aisy.set_datasets_root_folder("D:/traces/")
aisy.set_dataset("ascad-variable.h5")
aisy.set_database_name("database_ascad.sqlite")
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_number_of_profiling_traces(10000)
aisy.set_number_of_attack_traces(1000)
aisy.set_batch_size(400)
aisy.set_epochs(50)
aisy.set_neural_network(mlp)

param1 = [1, 2, 3]
param2 = "my_string"

custom_callbacks = [
    {
        "class": CustomCallback1,
        "name": "CustomCallback1",
        "parameters": [param1, param2]
    },
    {
        "class": CustomCallback2,
        "name": "CustomCallback2",
        "parameters": []
    }
]

aisy.run(
    callbacks=custom_callbacks
)

custom_callbacks = aisy.get_custom_callbacks()

custom_callback1 = custom_callbacks["CustomCallback1"]
print(custom_callback1.get_param1())
print(custom_callback1.get_param2())

custom_callback2 = custom_callbacks["CustomCallback2"]
