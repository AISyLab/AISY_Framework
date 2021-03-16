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

early_stopping = {
    "metrics": {
        # "accuracy": {
        #     "direction": "max",
        #     "class": "custom.custom_metrics.accuracy",
        #     "parameters": []
        # },
        # "loss": {
        #     "direction": "min",
        #     "class": "custom.custom_metrics.loss",
        #     "parameters": []
        # },
        "number_of_traces": {
            "direction": "min",
            "class": "custom.custom_metrics.number_of_traces",
            "parameters": []
        },
        # "success_rate": {
        #     "direction": "max",
        #     "class": "custom.custom_metrics.success_rate",
        #     "parameters": []
        # }
    }
}

aisy.run(
    early_stopping=early_stopping,
    key_rank_attack_traces=500
)

metrics_validation = aisy.get_metrics_validation()
for metric in metrics_validation:
    print("{}: {}".format(metric['metric'], metric['values']))
