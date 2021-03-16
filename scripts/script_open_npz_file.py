from app import *
import numpy as np

npz_file = np.load("{}npz/aes_attack.npz".format(resources_root_folder), allow_pickle=True)

# obtaining guessing entropy
guessing_entropy = npz_file["guessing_entropy"]
print(guessing_entropy)

# obtaining success_rate
success_rate = npz_file["success_rate"]
print(success_rate)

# obtaining metric for profiling set (accuracy, loss)
metrics_profiling = npz_file["metrics_profiling"]
print(metrics_profiling)

# obtaining metric for attack set (val_accuracy, val_loss)
metrics_attack = npz_file["metrics_attack"]
print(metrics_attack)

# obtaining metric for validation set (val_accuracy, val_loss and custom metrics)
metrics_validation = npz_file["metrics_validation"]
print(metrics_validation)

# obtaining trained weights from model
model_weights = npz_file["model_weights"]
print(model_weights)

# obtaining input gradient for each trained epoch (when visualization feature is set)
input_gradients_epoch = npz_file["input_gradients_epoch"]
print(input_gradients_epoch)

# obtaining sum of input gradient for all trained epochs (when visualization feature is set)
input_gradients_sum = npz_file["input_gradients_sum"]
print(input_gradients_sum)

# obtaining the analysis settings
settings = npz_file["settings"]
print(settings)

# obtaining hyperparameters
hyperparameters = npz_file["hyperparameters"]
print(hyperparameters)

# obtaining leakage model
leakage_model = npz_file["leakage_model"]
print(leakage_model)
