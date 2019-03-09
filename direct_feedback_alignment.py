import tensorflow as tf
from neuralnetwork import NeuralNetwork
from utils import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class DFANeuralNetwork(NeuralNetwork):
    def build(self):
        a = self.features
        for i in range(1, self.num_layers):
            _, a = self.fully_connected_layer(a, self.sizes[i], f"{self.scope}_layer{i}")

        self.acct_mat = tf.equal(tf.argmax(a, 1), tf.argmax(self.labels, 1))
        self.acct_res = tf.reduce_sum(tf.cast(self.acct_mat, tf.float32))

        error = tf.subtract(a, self.labels)
        a = self.features
        self.step = []
        for i in range(1, self.num_layers):
            a, weights_update, biases_update = self.direct_feedback_alignment(
                error, a, f"{self.scope}_layer{i}", i == self.num_layers - 1)
            self.step.append((weights_update, biases_update))

    def direct_feedback_alignment(self, output_error, input_act, scope, last_layer=False):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            z = tf.add(tf.matmul(input_act, weights), biases)
            a = tf.sigmoid(z)
            if not last_layer:
                random_weights = tf.get_variable("random_weights", [output_error.shape[1], weights.shape[1]],
                    initializer=tf.random_normal_initializer())
                error_directed = tf.matmul(output_error, random_weights)
            else:
                error_directed = output_error
            error = tf.multiply(error_directed, sigmoid_prime(z))
            delta_biases = error
            delta_weights = tf.matmul(tf.transpose(input_act), error)
            weights = tf.assign(weights, tf.subtract(weights, tf.multiply(self.learning_rate, delta_weights)))
            biases = tf.assign(biases, tf.subtract(biases, tf.multiply(self.learning_rate,
                tf.reduce_mean(delta_biases, axis=[0]))))
            return a, weights, biases


if __name__ == '__main__':
    DFA = DFANeuralNetwork([("-", 784), ("f", 50), ("a", 50), ("f", 10), ("a", 10)], scope='DFA')
    DFA.train()
