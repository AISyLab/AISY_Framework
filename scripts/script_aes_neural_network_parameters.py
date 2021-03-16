import aisy_sca
from app import *
from custom.custom_models.neural_networks import *


def mlp(classes, number_of_samples, neuron, layer, activation, learning_rate):
    model = Sequential(name="my_mlp")
    for l_i in range(layer):
        if l_i == 0:
            model.add(Dense(neuron, activation=activation, input_shape=(number_of_samples,)))
        else:
            model.add(Dense(neuron, activation=activation))
    model.add(Dense(classes, activation=None))
    model.add(Activation(activation="softmax"))
    model.summary()
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


aisy = aisy_sca.Aisy()
aisy.set_resources_root_folder(resources_root_folder)
aisy.set_database_root_folder(databases_root_folder)
aisy.set_datasets_root_folder(datasets_root_folder)
aisy.set_database_name("database_ascad.sqlite")
aisy.set_dataset(datasets_dict["ascad-variable.h5"])
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
aisy.set_epochs(40)
my_mlp = mlp(9, 1400, 200, 6, "relu", 0.001)
aisy.set_neural_network(my_mlp)
aisy.run()
