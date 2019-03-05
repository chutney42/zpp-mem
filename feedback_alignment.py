import tensorflow as tf
from neuralnetwork import NeuralNetwork
from utils import *


class FANeuralNetwork(NeuralNetwork):
    def backpropagation_fully_conneted(self, input_error, z, a, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights")
            random_weights = tf.get_variable("random_weights", shape=tf.transpose(weights).get_shape().as_list(),
                initializer=tf.random_normal_initializer())
            biases = tf.get_variable("biases")
            error = tf.multiply(input_error, sigmoid_prime(z))
            delta_biases = error
            delta_weights = tf.matmul(tf.transpose(a), error)
            output_error = tf.matmul(error, random_weights)
            weights = tf.assign(weights, tf.subtract(weights, tf.multiply(self.learning_rate, delta_weights)))
            biases = tf.assign(biases, tf.subtract(biases, tf.multiply(self.learning_rate, tf.reduce_mean(delta_biases,
                axis=[0]))))
            return output_error, weights, biases


if __name__ == '__main__':
    FA = FANeuralNetwork([("-", 784), ("f", 50), ("a", 50), ("f", 10), ("a", 10)], scope="FA")
    FA.train()
