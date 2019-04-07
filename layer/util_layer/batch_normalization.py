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

    def build_forward(self, input_vec, remember_input=True, gather_stats=False):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

            self.save_shape(input_vec)
            if remember_input:
                self.input_vec = input_vec
            input_shape=input_vec.get_shape()[1:]

            gamma = tf.get_variable("gamma", input_shape, initializer=tf.ones_initializer())
            beta = tf.get_variable("beta", input_shape, initializer=tf.zeros_initializer())
            batch_mean, batch_var = tf.nn.moments(input_vec, [0])

            tf.nn.batch_normalization(input_vec, batch_mean, batch_var, beta, gamma, self.epsilon, "batch_n")

            input_act_normalized = (input_vec - batch_mean) / tf.sqrt(batch_var + self.epsilon)
            input_act_normalized = gamma * input_act_normalized + beta

            if gather_stats:
                print(f"ahaha {input_vec}")
                tf.summary.histogram("for_input_not_normalized", input_vec)
                tf.summary.histogram("for_var", batch_var)
                tf.summary.histogram("for_mean", batch_mean)
                tf.summary.histogram("for_input_normalized", input_act_normalized)

            self.output = input_act_normalized

            return input_act_normalized

    def build_backward(self, error, gather_stats=False):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_vec = self.restore_input()

            gamma = tf.get_variable("gamma")
            beta = tf.get_variable("beta")

            grads = tf.gradients(self.output, [input_vec,gamma, beta], error)

            update_beta = tf.assign(beta, tf.subtract(beta, tf.multiply(grads[2], self.learning_rate)))
            update_gamma = tf.assign(gamma, tf.subtract(gamma, tf.multiply(grads[1], self.learning_rate)))
            self.step = (update_beta, update_gamma)

            if gather_stats:
                print(f"ahaha {input_vec}")
                tf.summary.histogram("input_not_normalized", input_vec)
                tf.summary.histogram("gamma", gamma)
                tf.summary.histogram("beta", beta)
                tf.summary.histogram("input_normalized", self.output)
                tf.summary.histogram("d_gamma", grads[1])
                tf.summary.histogram("d_beta", grads[2])
                tf.summary.histogram("output_error", grads[0])

            return grads[0]
