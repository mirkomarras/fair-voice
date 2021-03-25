import tensorflow as tf


def fair_loss(classes):
    def loss(y_true, y_pred):
        sens_attr = tf.map_fn(lambda g: g == 1, y_true[:, :classes][:, 0], dtype=tf.bool)
        y_true = y_true[:, classes:]

        y_t_male = tf.gather(y_true, tf.reshape(tf.where(sens_attr), [-1]))
        y_p_male = tf.gather(y_pred, tf.reshape(tf.where(sens_attr), [-1]))
        not_sens_attr = tf.math.logical_not(sens_attr)
        y_t_female = tf.gather(y_true, tf.reshape(tf.where(not_sens_attr), [-1]))
        y_p_female = tf.gather(y_pred, tf.reshape(tf.where(not_sens_attr), [-1]))

        cc_male = tf.keras.losses.categorical_crossentropy(y_t_male, y_p_male)
        cc_female = tf.keras.losses.categorical_crossentropy(y_t_female, y_p_female)

        cc_male = tf.keras.backend.mean(cc_male)
        cc_female = tf.keras.backend.mean(cc_female)
        return tf.keras.backend.square(cc_male - cc_female)

    return loss
