import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from utils import *

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class NeuralNetwork(object):
    def __init__(self, layers, learning_rate=0.5, scope="main"):
        self.scope = scope
        self.num_layers = len(layers)
        self.types = [a[0] for a in layers]
        self.sizes = [a[1] for a in layers]
        self.labels = tf.placeholder(tf.float32, [None, self.sizes[-1]])
        self.learning_rate = tf.constant(learning_rate)
        self.features = tf.placeholder(tf.float32, [None, self.sizes[0]])
        self.vars = [self.features]
        self.step = []

    def build(self):
        def normalize(i):
            return self.types[i] == "n"

        def activate(i):
            return self.types[i] == "a"

        def fully_connect(i):
            return self.types[i] == "f"

        a = self.features
        # FORWARD PASS
        for i in range(1, self.num_layers):
            if fully_connect(i):
                self.vars.append(a)
                a = self.fully_connected_layer(a, self.sizes[i], '{}_layer{}'.format(self.scope, i))
            elif activate(i):
                self.vars.append(a)
                a = self.activate_layer(a, tf.sigmoid, '{}_layer{}'.format(self.scope, i))
            elif normalize(i):
                self.vars.append(a)
                a = self.normalization_layer(a, self.sizes[i], '{}_layer{}'.format(self.scope, i))
            else:
                raise Exception('{} is not recognized layer type'.format(self.types[i]))

        output = a
        upstream_error = tf.subtract(output, self.labels)

        # BACKWARD PASS
        for i in range(len(self.vars) - 1, -1, -1):
            if fully_connect(i):
                a = self.vars[i]
                upstream_error, w_update, b_update = \
                    self.backpropagation_fully_connected(upstream_error, a, '{}_layer{}'.format(self.scope, i))
                self.step.append((w_update, b_update))
            elif activate(i):
                a = self.vars[i]
                upstream_error = \
                    self.backpropagation_activate(upstream_error, a, '{}_layer{}'.format(self.scope, i))
            elif normalize(i):
                a = self.vars[i]
                upstream_error, gamma_update, beta_update = \
                    self.backprogation_normalization(upstream_error, a, self.sizes[i],
                                                     '{}_layer{}'.format(self.scope, i))

                self.step.append((beta_update, gamma_update))
                upstream_error = self.backpropagation_activate(upstream_error, a, '{}_layer{}'.format(self.scope, i))
            else:
                raise Exception('{} is not recognized layer type, also how did you reach this code???'.format(self.types[i]))

        self.acct_mat = tf.equal(tf.argmax(output, 1), tf.argmax(self.labels, 1))
        self.acct_res = tf.reduce_sum(tf.cast(self.acct_mat, tf.float32))

    def fully_connected_layer(self, input_act, output_dim, scope):
        with tf.variable_scope(scope):
            weights = tf.get_variable("weights", [input_act.shape[1], output_dim],
                                      initializer=tf.random_normal_initializer())
            biases = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer())
            tf.summary.histogram("weights:", weights)
            tf.summary.histogram("biases:", biases)

            output_act = tf.add(tf.matmul(input_act, weights), biases)
            return output_act

    def activate_layer(self, input_act, act_func, scope):
        with tf.variable_scope(scope):
            output_act = act_func(input_act)
            return output_act

    def backpropagation_fully_connected(self, upstream_error, layer_input, scope):
        with tf.variable_scope(scope, reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            roc_cost_over_biases = upstream_error
            roc_cost_over_weights = tf.matmul(tf.transpose(layer_input), upstream_error)
            output_error = tf.matmul(upstream_error, tf.transpose(weights))  # da lower

            weights_step = tf.scalar_mul(self.learning_rate, roc_cost_over_weights)
            weights_update = tf.assign(weights, tf.subtract(weights, weights_step))

            biases_step = tf.scalar_mul(self.learning_rate, tf.reduce_mean(roc_cost_over_biases, axis=[0]))
            biases_update = tf.assign(biases, tf.subtract(biases, biases_step))

            return output_error, weights_update, biases_update

    def backpropagation_activate(self, upstream_error, layer_input, scope):
        with tf.variable_scope(scope, reuse=True):
            output_error = tf.multiply(upstream_error, sigmoid_prime(layer_input))
            return output_error

    def backprogation_normalization(self, upstream_error, layer_input, layer_output_dim, scope):
        with tf.variable_scope(scope, reuse=True):
            epsilon = 0.001  # In case batch_var=0, division by zero is not cool
            gamma = tf.get_variable("gamma")
            beta = tf.get_variable("beta")
            N = layer_output_dim
            batch_mean, batch_var = tf.nn.moments(layer_input, [0])

            z_normalized = (layer_input - batch_mean) / tf.sqrt(batch_var + epsilon)

            z_mu = layer_input - batch_mean
            std_inv = 1 / tf.sqrt(tf.add(batch_var, epsilon))
            dz_norm = upstream_error * gamma
            dvar = tf.matmul(tf.scalar_mul(-.5, tf.reduce_sum(tf.matmul(dz_norm, z_mu), 0)), tf.pow(std_inv, 3))
            dmu = tf.add(tf.reduce_sum(tf.matmul(dz_norm, -std_inv), 0),
                         tf.matmul(dvar, tf.reduce_mean(tf.scalar_mul(-2., z_mu), 0)))

            output_error = tf.add(tf.add(tf.matmul(dz_norm, std_inv), tf.matmul(tf.scalar_mul(2. / N, dvar), z_mu)),
                        tf.scalar_mul(1. / N, dmu))

            dgamma = tf.reduce_sum(tf.matmul(upstream_error, z_normalized), 0)
            dbeta = tf.reduce_sum(upstream_error, 0)

            return output_error, \
                   tf.assign(gamma, tf.subtract(gamma, tf.multiply(self.learning_rate, dgamma))), \
                   tf.assign(beta, tf.subtract(dbeta, tf.multiply(self.learning_rate, dbeta)))

    def normalization_layer(self, input_act, y_dim, scope):
        with tf.variable_scope(scope):
            epsilon = 0.001  # In case batch_var=0, division by zero is not cool
            gamma = tf.get_variable("gamma", [y_dim], initializer=tf.ones_initializer())
            beta = tf.get_variable("beta", [y_dim], initializer=tf.ones_initializer())

            batch_mean, batch_var = tf.nn.moments(input_act, [0])
            input_act_normalized = (input_act - batch_mean) / tf.sqrt(batch_var + epsilon)

            input_act_normalized = gamma * input_act_normalized + beta
            return input_act_normalized

    def train(self, batch_size=10, batch_num=100000):
        # TODO generlize it to other datasets using tf.data for example
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("logs/", sess.graph)
            sess.run(tf.global_variables_initializer())
            for i in range(batch_num):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(self.step, feed_dict={self.features: batch_xs, self.labels: batch_ys})
                if i % 1000 == 0:
                    res = sess.run(self.acct_res, feed_dict={self.features: mnist.test.images[:1000], self.labels:
                        mnist.test.labels[:1000]})
                    print("{}%".format(res / 10))
            res = sess.run(self.acct_res, feed_dict={self.features: mnist.test.images, self.labels: mnist.test.labels})
            print("{}%".format(res / len(mnist.test.labels) * 100))
            # TODO save model
            writer.close()

    def infer(self, x):
        # TODO restore model
        with tf.Session() as sess:
            res = sess.run(self.vars[-1], feed_dict={self.features: x})
        return res

    def test(self, x, y):
        # TODO restore model
        with tf.Session() as sess:
            res = sess.run(self.acct_res, feed_dict={self.features: x, self.labels: y})
        return res


if __name__ == '__main__':
    NN = NeuralNetwork([("-", 784), ("f", 50), ("a", 50), ("f", 10), ("a", 10)])
    # NN = NeuralNetwork([("-", 784), ("f", 50), ("n", 50),("a", 50), ("f", 10), ("a", 10)])
    NN.build()
    NN.train()
