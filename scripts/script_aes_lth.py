from custom.custom_models.neural_networks import *
from aisy.sca_deep_learning_aes import AisyAes
from lottery_ticket_pruner import LotteryTicketPruner

aisy = AisyAes()
aisy.set_dataset("ascad-variable.h5")
aisy.set_database_name("database_ascad.sqlite")
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_number_of_profiling_traces(100000)
aisy.set_number_of_attack_traces(2000)
aisy.set_batch_size(400)
aisy.set_epochs(1)
aisy.set_neural_network(mlp)

initial_weights = aisy.get_model().get_weights()

param1 = 0
custom_callbacks = [
    {
        "class": "SaveWeights",
        "parameters": [param1]
    }
]

aisy.run(
    callbacks=custom_callbacks,
    visualization=[5000]
)

custom_callbacks = aisy.get_custom_callbacks()

callback_weights = custom_callbacks["SaveWeights"]
trained_weights = callback_weights.get_weights()[0]

aisy.set_neural_network(mlp)
model = aisy.get_model()

model.set_weights(initial_weights)
lth_pruner = LotteryTicketPruner(model)

# ----------------------------------------------------------------------------------------------------------------------------------
# 2.3. Re-initialize and train Pruned Model with initial weights from Baseline Model (Lottery Ticket)
# ----------------------------------------------------------------------------------------------------------------------------------
sparsity_level = 90

model.set_weights(trained_weights)
lth_pruner.set_pretrained_weights(model)
lth_pruner.calc_prune_mask(model, 0.01 * sparsity_level, 'smallest_weights')
model.set_weights(initial_weights)
lth_pruner.apply_pruning(model)

aisy = AisyAes()
aisy.set_dataset("ascad-variable.h5")
aisy.set_database_name("database_ascad.sqlite")
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_number_of_profiling_traces(100000)
aisy.set_number_of_attack_traces(1000)
aisy.set_batch_size(400)
aisy.set_epochs(20)
aisy.set_neural_network(mlp)
aisy.set_model_weights(model.get_weights())

pruner = lth_pruner

custom_callbacks = [
    {
        "class": "PrunerCallback",
        "name": "PrunerCallback",
        "parameters": [pruner]
    }
]

aisy.run(
    callbacks=custom_callbacks,
    visualization=[5000]
)
