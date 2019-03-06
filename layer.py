import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import numpy as np
from utils import *

class Layer(object):
    def __init__(self, scope="layer"):
        self.scope = scope

class FullyConnected(Layer):
    def __init__(self, output_dim, trainable=True, learning_rate=0.5, scope="fully_connected_layer"):
        self.output_dim = output_dim
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.scope = scope

    def build_forward(self, input_vec, remember_input=True):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights", [input_vec.shape[1], self.output_dim],
                initializer=tf.random_normal_initializer())
            biases = tf.get_variable("biases", [self.output_dim],
                initializer=tf.constant_initializer())
            z = tf.add(tf.matmul(input_vec, weights), biases)
            if remember_input:
                self.input_vec = input_vec
            return z

    def build_optimize(self, error, input_vec=None):
        if not input_vec:
            input_vec = self.input_vec
        self.input_vec = None
        with tf.variable_scope(self.scope, reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            delta_biases = error
            delta_weights = tf.matmul(tf.transpose(input_vec), error)
            bp_error = tf.matmul(error, tf.transpose(weights))
            weights = tf.assign(weights, tf.subtract(weights, tf.multiply(self.learning_rate, delta_weights)))
            biases = tf.assign(biases, tf.subtract(biases, tf.multiply(self.learning_rate, tf.reduce_mean(delta_biases,
                axis=[0]))))
            self.step = (weights, biases)
            return bp_error

class FAFullyConnected(FullyConnected):
    def build_optimize(self, error, input_vec=None):
        if not input_vec:
            input_vec = self.input_vec
        self.input_vec = None
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights")
            random_weights = tf.get_variable("random_weights", shape=tf.transpose(weights).get_shape().as_list(),
                initializer=tf.random_normal_initializer())
            biases = tf.get_variable("biases")
            delta_biases = error
            delta_weights = tf.matmul(tf.transpose(input_vec), error)
            bp_error = tf.matmul(error, random_weights)
            weights = tf.assign(weights, tf.subtract(weights, tf.multiply(self.learning_rate, delta_weights)))
            biases = tf.assign(biases, tf.subtract(biases, tf.multiply(self.learning_rate, tf.reduce_mean(delta_biases,
                axis=[0]))))
            self.step = (weights, biases)
            return bp_error

class DFAFullyConnected(FullyConnected):
    def build_direct(self, error, last_layer=False):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if not last_layer:
                random_weights = tf.get_variable("random_weights", [error.shape[1], self.output_dim], initializer=tf.random_normal_initializer())
                dfa_error = tf.matmul(error, random_weights)
            else:
                dfa_error = error
            return dfa_error

    def build_optimize(self, error, input_vec=None):
        if not input_vec:
            input_vec = self.input_vec
        self.input_vec = None
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            delta_biases = error
            delta_weights = tf.matmul(tf.transpose(input_vec), error)
            weights = tf.assign(weights, tf.subtract(weights, tf.multiply(self.learning_rate, delta_weights)))
            biases = tf.assign(biases, tf.subtract(biases, tf.multiply(self.learning_rate, tf.reduce_mean(delta_biases,
                axis=[0]))))
            self.step = (weights, biases)
            return None

class DFABlock(Layer):
    def __init__(self, sequence, trainable=True, scope="dfa_block_layer"):
        self.sequence = sequence
        self.trainable = trainable
        self.scope = scope

    def build_forward(self, input_vec):
        for i, layer in enumerate(self.sequence):
            layer.scope = '{}_{}_{}'.format(self.scope, layer.scope, i)
            input_vec = layer.build_forward(input_vec, remember_input=False)
        return input_vec

    def build_optimize(self, input_vec, error, last_layer):
        for i, layer in enumerate(self.sequence):
            input_vec = layer.build_forward(input_vec, remember_input=True)
        error = self.sequence[0].build_direct(error, last_layer)
        self.step = []
        for i, layer in reversed(list(enumerate(self.sequence))):
            error = layer.build_optimize(error)
            if layer.trainable:
                self.step.append(layer.step)
        return input_vec

class ActivationFunction(Layer):
    def __init__(self, func, func_prime, scope="activation_function_layer"):
        self.func = func
        self.func_prime = func_prime
        self.trainable = False
        self.scope = scope

    def build_forward(self, input_vec, remember_input=True):
        if remember_input:
            self.input_vec = input_vec
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            return self.func(input_vec)

    def build_optimize(self, error, input_vec=None):
        if not input_vec:
            input_vec = self.input_vec
        self.input_vec = None
        with tf.variable_scope(self.scope):
            return tf.multiply(error, self.func_prime(input_vec))

class Sigmoid(ActivationFunction):
    def __init__(self, scope="sigmoid_layer"):
        super().__init__(tf.sigmoid, tf_sigmoid_prime, scope)

class BatchNormalization(Layer):
    def __init__(self, trainable=True, learning_rate=0.5, scope="batch_normalization_layer"):
        self.trainable = trainable
        self.scope = scope
        self.epsilon = 0.0000001
        self.step = None
        self.learning_rate = learning_rate
        self.input_vec = None

    def build_forward(self, input_vec, remember_input=True):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            N = input_vec.get_shape()[1]
            gamma = tf.get_variable("gamma", [N], initializer=tf.ones_initializer())
            beta = tf.get_variable("beta", [N], initializer=tf.zeros_initializer())
            tf.summary.histogram("input_not_normalized:", input_vec)
            batch_mean, batch_var = tf.nn.moments(input_vec, [0])
            tf.summary.histogram("var:", batch_var)
            tf.summary.histogram("mean:", batch_mean)
            input_act_normalized = (input_vec - batch_mean) / tf.sqrt(batch_var + self.epsilon)

            input_act_normalized = gamma * input_act_normalized + beta
            tf.summary.histogram("input_normalized:", input_act_normalized)
            if remember_input:
                self.input_vec = input_vec
            return input_act_normalized

    def build_optimize(self, error, input_vec=None):
        if not input_vec:
            input_vec = self.input_vec
        self.input_vec = None
        with tf.variable_scope(self.scope, reuse=True):
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
