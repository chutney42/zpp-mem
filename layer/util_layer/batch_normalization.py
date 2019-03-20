import tensorflow as tf
from functools import reduce

from layer.layer import Layer


class BatchNormalization(Layer):
    def __init__(self, learning_rate=0.5, scope="batch_normalization_layer"):
        super().__init__(trainable=True, scope=scope)
        self.epsilon = 0.0000001
        self.learning_rate = learning_rate

    def __str__(self):
        return "BatchNormalization()"

    def build_forward(self, input_vec, remember_input=True, gather_stats=True):
        self.save_shape(input_vec)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if remember_input:
                self.input_vec = input_vec
            input_shape=input_vec.get_shape()[1:]

            gamma = tf.get_variable("gamma", input_shape, initializer=tf.ones_initializer())
            beta = tf.get_variable("beta", input_shape, initializer=tf.zeros_initializer())
            batch_mean, batch_var = tf.nn.moments(input_vec, [0])

            input_act_normalized = (input_vec - batch_mean) / tf.sqrt(batch_var + self.epsilon)
            input_act_normalized = gamma * input_act_normalized + beta

            if gather_stats:
                tf.summary.histogram("input_not_normalized", input_vec)
                tf.summary.histogram("var", batch_var)
                tf.summary.histogram("mean", batch_mean)
                tf.summary.histogram("input_normalized", input_act_normalized)

            return input_act_normalized

    def build_backward(self, error, gather_stats=True):
        input_vec = self.restore_input()
        with tf.variable_scope(self.scope, reuse=True):
            nm_of_channels = int(reduce(lambda x, y: x * y, input_vec.shape[1:]))  # Number of features in input_vec

            gamma = tf.get_variable("gamma")
            beta = tf.get_variable("beta")
            batch_mean, batch_var = tf.nn.moments(input_vec, [0])

            input_act_normalized = (input_vec - batch_mean) / tf.sqrt(batch_var + self.epsilon)

            layer_input_zeroed = input_vec - batch_mean
            std_inv = 1. / tf.sqrt(batch_var + self.epsilon)

            dz_norm = error * gamma
            dvar = -0.5 * tf.reduce_sum(tf.multiply(dz_norm, layer_input_zeroed), 0) * tf.pow(std_inv, 3)

            dmu = tf.reduce_sum(dz_norm * -std_inv, [0]) + dvar * tf.reduce_mean(-2. * layer_input_zeroed, [0])

            output_error = dz_norm * std_inv + (dvar * 2 * layer_input_zeroed / nm_of_channels) + dmu / nm_of_channels
            dgamma = tf.reduce_sum(tf.multiply(error, input_act_normalized), [0])
            dbeta = tf.reduce_sum(error, [0])
            update_beta = tf.assign(beta, tf.subtract(beta, tf.multiply(dbeta, self.learning_rate)))
            update_gamma = tf.assign(gamma, tf.subtract(gamma, tf.multiply(dgamma, self.learning_rate)))
            self.step = (update_beta, update_gamma)
            return output_error
