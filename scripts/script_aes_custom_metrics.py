from custom.custom_models.neural_networks import *
from commons.sca_aisy_aes import Aisy

aisy = Aisy()
aisy.set_datasets_root_folder("D:/traces/")
aisy.set_dataset("ascad-variable.h5")
aisy.set_database_name("database_ascad_early_stopping.sqlite")
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_number_of_profiling_traces(100000)
aisy.set_number_of_attack_traces(2000)
aisy.set_batch_size(400)
aisy.set_epochs(25)
aisy.set_neural_network(mlp)

early_stopping = {
    "metrics": {
        "accuracy": {
            "direction": "max",
            "class": "accuracy",
            "parameters": []
        },
        "loss": {
            "direction": "min",
            "class": "loss",
            "parameters": []
        },
        "number_of_traces": {
            "direction": "min",
            "class": "number_of_traces",
            "parameters": []
        },
        "success_rate": {
            "direction": "max",
            "class": "success_rate",
            "parameters": []
        }
    }
}

aisy.run(
    early_stopping=early_stopping,
    key_rank_attack_traces=1000
)

metrics_validation = aisy.get_metrics_validation()
for metric in metrics_validation:
    print("{}: {}".format(metric['metric'], metric['values']))
