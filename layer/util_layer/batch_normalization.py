import tensorflow as tf

from layer.layer import Layer


class BatchNormalization(Layer):
    def __init__(self, learning_rate=0.5, scope="batch_normalization_layer"):
        super().__init__(trainable=True, scope=scope)
        self.epsilon = 0.000001
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

            input_act_normalized = tf.nn.batch_normalization(input_vec, batch_mean, batch_var, beta, gamma, self.epsilon, "batch_n")

            self.output = input_act_normalized

            if gather_stats:
                tf.summary.histogram("mean", batch_mean, family=self.scope)
                tf.summary.histogram("var", batch_var, family=self.scope)

            return input_act_normalized

    def build_backward(self, error, gather_stats=False):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_vec = self.restore_input()

            gamma = tf.get_variable("gamma")
            beta = tf.get_variable("beta")

            grads = tf.gradients(self.output, [input_vec, gamma, beta], error)

            update_gamma = tf.assign(gamma, tf.subtract(gamma, tf.multiply(grads[1], self.learning_rate)))
            update_beta = tf.assign(beta, tf.subtract(beta, tf.multiply(grads[2], self.learning_rate)))
            self.step = (update_beta, update_gamma)

            if gather_stats:
                tf.summary.histogram("input", input_vec, family=self.scope)
                tf.summary.histogram("gamma", gamma, family=self.scope)
                tf.summary.histogram("beta", beta, family=self.scope)
                tf.summary.histogram("input_normalized", self.output, family=self.scope)
                tf.summary.histogram("delta_gamma", grads[1], family=self.scope)
                tf.summary.histogram("delta_beta", grads[2], family=self.scope)
                tf.summary.histogram("output_error", grads[0], family=self.scope)

            return grads[0]
