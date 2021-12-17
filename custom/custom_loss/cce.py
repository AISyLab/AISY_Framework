import tensorflow.keras as tk


def categorical_cross_entropy_loss(settings, model_args=None, parameters=None):
    def loss(y_true, y_pred):
        """ For BO, remove it for release """
        y_true = y_true[:, :settings["classes"]]
        return tk.backend.categorical_crossentropy(y_true, y_pred)

    return loss
