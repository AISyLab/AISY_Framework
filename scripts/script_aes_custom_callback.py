import aisy_sca
from app import *
from custom.custom_models.neural_networks import *

aisy = aisy_sca.Aisy()
aisy.set_resources_root_folder(resources_root_folder)
aisy.set_database_root_folder(databases_root_folder)
aisy.set_datasets_root_folder(datasets_root_folder)
aisy.set_database_name("database_ascad.sqlite")
aisy.set_dataset(datasets_dict["ascad-variable.h5"])
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
aisy.set_epochs(10)
aisy.set_neural_network(mlp)

custom_callbacks = [
    {
        "class": "custom.custom_callbacks.callbacks.CustomCallback1",
        "name": "CustomCallback1",
        "parameters": {
            "param1": [1, 2, 3],
            "param2": "my_string"
        }
    },
    {
        "class": "custom.custom_callbacks.callbacks.CustomCallback2",
        "name": "CustomCallback2",
        "parameters": {}
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
