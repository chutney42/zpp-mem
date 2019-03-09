import tensorflow as tf
from utils import *


class Layer(object):
    def __init__(self, scope="layer"):
        self.scope = scope

    def build_forward(self, input_vec, gather_stats=True):
        raise NotImplementedError("This method should be implemented in subclass")

    def build_backward(self, error, gather_stats=True):
        raise NotImplementedError("This method should be implemented in subclass")


class FullyConnected(Layer):
    def __init__(self, output_dim, trainable=True, learning_rate=0.5, scope="fully_connected_layer"):
        super().__init__(scope)
        self.output_dim = output_dim
        self.trainable = trainable
        self.learning_rate = learning_rate

    def build_forward(self, input_vec, gather_stats=True):
        with tf.variable_scope(self.scope):
            weights = tf.get_variable("weights", [input_vec.shape[1], self.output_dim],
                                      initializer=tf.random_normal_initializer())
            biases = tf.get_variable("biases", [self.output_dim],
                                     initializer=tf.constant_initializer())
            z = tf.add(tf.matmul(input_vec, weights), biases)
            self.input_vec = input_vec
            return z

    def build_backward(self, error, gather_stats=True):
        with tf.variable_scope(self.scope, reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            roc_cost_over_biases = error
            roc_cost_over_weights = tf.matmul(tf.transpose(self.input_vec), error)
            bp_error = tf.matmul(error, tf.transpose(weights))
            weights = tf.assign(weights, tf.subtract(weights, tf.multiply(self.learning_rate, roc_cost_over_weights)))
            biases = tf.assign(biases,
                               tf.subtract(biases, tf.multiply(self.learning_rate, tf.reduce_mean(roc_cost_over_biases,
                                                                                                  axis=[0]))))
            self.step = (weights, biases)
            return bp_error


class ActivationFunction(Layer):
    def __init__(self, func, func_prime, scope="activation_function_layer"):
        super().__init__(scope)
        self.func = func
        self.func_prime = func_prime
        self.trainable = False

    def build_forward(self, input_vec, gather_stats=True):
        with tf.variable_scope(self.scope):
            self.input_vec = input_vec
            return self.func(input_vec)

    def build_backward(self, error, gather_stats=True):
        with tf.variable_scope(self.scope):
            return tf.multiply(error, self.func_prime(self.input_vec))


class Sigmoid(ActivationFunction):
    def __init__(self, scope="sigmoid_layer"):
        super().__init__(tf.sigmoid, sigmoid_prime, scope)


class BatchNormalization(Layer):
    def __init__(self, trainable=True, learning_rate=0.5, scope="batch_normalization_layer"):
        super().__init__(scope)
        self.trainable = trainable
        self.epsilon = 0.0000001
        self.step = None
        self.learning_rate = learning_rate
        self.input_vec = None

    def build_forward(self, input_vec, gather_stats=True):
        with tf.variable_scope(self.scope):
            self.input_vec = input_vec
            N = input_vec.get_shape()[1]
            gamma = tf.get_variable("gamma", [N], initializer=tf.ones_initializer())
            beta = tf.get_variable("beta", [N], initializer=tf.zeros_initializer())
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
        with tf.variable_scope(self.scope, reuse=True):
            input_vec = self.input_vec
            N = int(input_vec.get_shape()[1])
            gamma = tf.get_variable("gamma")
            beta = tf.get_variable("beta")
            batch_mean, batch_var = tf.nn.moments(input_vec, [0])

            input_act_normalized = (input_vec - batch_mean) / tf.sqrt(batch_var + self.epsilon)

            layer_input_zeroed = input_vec - batch_mean
            std_inv = 1. / tf.sqrt(batch_var + self.epsilon)
            dz_norm = error * gamma
            dvar = -0.5 * tf.reduce_sum(tf.multiply(dz_norm, layer_input_zeroed), 0) * tf.pow(std_inv, 3)

            dmu = tf.reduce_sum(dz_norm * -std_inv, [0]) + dvar * tf.reduce_mean(-2. * layer_input_zeroed, [0])

            output_error = dz_norm * std_inv + (dvar * 2 * layer_input_zeroed / N) + dmu / N

            dgamma = tf.reduce_sum(tf.multiply(error, input_act_normalized), [0])
            dbeta = tf.reduce_sum(error, [0])
            update_beta = tf.assign(beta, tf.subtract(beta, tf.multiply(dbeta, self.learning_rate)))
            update_gamma = tf.assign(gamma, tf.subtract(gamma, tf.multiply(dgamma, self.learning_rate)))
            self.step = (update_beta, update_gamma)
            return output_error
