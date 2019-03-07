import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from utils import *


class Layer(object):
    def __init__(self, scope="layer"):
        self.trainable = True
        self.input_shape = None
        self.scope = scope
        self.learning_rate = None

    def save_shape(self, input_vec):
        self.input_shape = tf.shape(input_vec)

    def restore_shape(self, input_vec):
        return tf.reshape(input_vec, self.input_shape)

    def flatten_input(self, input_vec):
        return tf.layers.Flatten()(input_vec)


class ConvolutionalLayer(Layer):
    def __init__(self, filter_dim, stride=[1, 1, 1, 1], number_of_filters=1, padding="SAME",
                 trainable=True, learning_rate=0.5,
                 scope="convoluted_layer"):
        self.stride = stride
        self.filter_dim = filter_dim
        self.number_of_filters = number_of_filters
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.scope = scope
        self.padding = padding

    def build_forward(self, input_matrix):
        with tf.variable_scope(self.scope):
            self.input_shape = tf.shape(input_matrix)
            self.input_matrix = input_matrix
            (width, length, depth) = input_matrix.shape[1], input_matrix.shape[2], input_matrix.shape[3]
            filter_shape = [self.filter_dim, self.filter_dim, depth,
                            self.number_of_filters]
            filters = tf.get_variable("filters", filter_shape,
                                      initializer=tf.random_normal_initializer())
            output = tf.nn.conv2d(input_matrix, filters, strides=self.stride, padding=self.padding, name="Convolution")
            return output

    def build_backward(self, error_matrix):
        with tf.variable_scope(self.scope, reuse=True):
            filters = tf.get_variable("filters")
            input_matrix = self.input_matrix

            backprop_error = tf.nn.conv2d_backprop_input(self.input_shape, filters, error_matrix, self.stride,
                                                         self.padding)

            roc_cost_over_filter = tf.nn.conv2d_backprop_filter(input_matrix, tf.shape(filters), error_matrix,
                                                                self.stride, self.padding)
            self.step = tf.assign(filters, filters - self.learning_rate * roc_cost_over_filter)
            return backprop_error


class FullyConnected(Layer):
    def __init__(self, output_dim, trainable=True, learning_rate=0.5, scope="fully_connected_layer"):
        self.output_dim = output_dim
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.scope = scope

    def build_forward(self, input_vec):
        with tf.variable_scope(self.scope):
            self.save_shape(input_vec)
            input_vec = self.flatten_input(input_vec)
            self.input_vec = input_vec
            weights = tf.get_variable("weights", [input_vec.shape[1], self.output_dim],
                                      initializer=tf.random_normal_initializer())
            biases = tf.get_variable("biases", [self.output_dim],
                                     initializer=tf.constant_initializer())
            z = tf.add(tf.matmul(input_vec, weights), biases)
            return z

    def build_backward(self, error):
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
            return self.restore_shape(bp_error)


class ActivationFunction(Layer):
    def __init__(self, func, func_prime, scope="activation_function_layer"):
        self.func = func
        self.func_prime = func_prime
        self.trainable = False
        self.scope = scope

    def build_forward(self, input_vec):
        with tf.variable_scope(self.scope):
            self.input_vec = input_vec
            return self.func(input_vec)

    def build_backward(self, error):
        with tf.variable_scope(self.scope):
            return tf.multiply(error, self.func_prime(self.input_vec))


class Sigmoid(ActivationFunction):
    def __init__(self, scope="sigmoid_layer"):
        super().__init__(tf.sigmoid, sigmoid_prime, scope)


class BatchNormalization(Layer):
    def __init__(self, trainable=True, learning_rate=0.5, scope="batch_normalization_layer"):
        self.trainable = trainable
        self.scope = scope
        self.epsilon = 0.0000001
        self.step = None
        self.learning_rate = learning_rate
        self.input_vec = None

    def build_forward(self, input_vec, gather_stats=True):
        with tf.variable_scope(self.scope):
            self.save_shape(input_vec)
            input_vec = self.flatten_input(input_vec)
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

            return self.restore_shape(input_act_normalized)

    def build_backward(self, error):
        with tf.variable_scope(self.scope, reuse=True):
            input_vec = self.input_vec
            error = self.flatten_input(error)

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
            return self.restore_shape(output_error)
