import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
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

    def build_forward(self, input_vec):
        with tf.variable_scope(self.scope):
            weights = tf.get_variable("weights", [input_vec.shape[1], self.output_dim],
                initializer=tf.random_normal_initializer())
            biases = tf.get_variable("biases", [self.output_dim],
                initializer=tf.constant_initializer())
            z = tf.add(tf.matmul(input_vec, weights), biases)
            self.input_vec = input_vec
            return z

    def build_backward(self, error):
        with tf.variable_scope(self.scope, reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            roc_cost_over_biases = error
            roc_cost_over_weights = tf.matmul(tf.transpose(self.input_vec), error)
            bp_error = tf.matmul(error, tf.transpose(weights))
            weights = tf.assign(weights, tf.subtract(weights, tf.multiply(self.learning_rate, roc_cost_over_weights)))
            biases = tf.assign(biases, tf.subtract(biases, tf.multiply(self.learning_rate, tf.reduce_mean(roc_cost_over_biases,
                axis=[0]))))
            self.step = (weights, biases)
            return bp_error

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
