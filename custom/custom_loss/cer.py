import tensorflow as tf


def cer_loss(settings,  model_args=None, parameters=None):

    n = parameters["n"]

    def cer(y_true, y_pred):

        """ For BO, remove it for release """
        y_true = y_true[:, :settings["classes"]]
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        ce_shuffled = 0.0

        for i in range(n):
            y_true_shuffled = tf.random.shuffle(y_true)
            ce_shuffled += tf.keras.losses.categorical_crossentropy(y_true_shuffled, y_pred)

        ce_shuffled = ce_shuffled / n

        return ce / ce_shuffled

    return cer
