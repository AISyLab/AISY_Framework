from custom.custom_models.neural_networks import *
from commons.sca_aisy_aes import Aisy


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


aisy = Aisy()
aisy.set_dataset("ascad-variable.h5")
aisy.set_database_name("database_ascad.sqlite")
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
aisy.set_epochs(50)
aisy.set_neural_network(mlp(9, 1400, 200, 3, "relu", 0.001))
aisy.run()
