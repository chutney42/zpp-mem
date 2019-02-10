import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from utils import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

file_name = "run_auto_increment"

class NeuralNetwork(object):
    def __init__(self, sizes, learning_rate=0.5, scope="main"):
        self.scope = scope
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.learning_rate = tf.constant(learning_rate)
        self.labels = tf.placeholder(tf.float32, [None, self.sizes[-1]])
        self.features = tf.placeholder(tf.float32, [None, self.sizes[0]])

        with open(file_name) as file:
            self.run_number = int(file.read())

        self.run_number += 1

        with open(file_name, 'w') as file:
            file.write(str(self.run_number))

    def build(self):
        self.acts = [self.features]
        self.zs = []
        a = self.acts[0]
        for i in range(1, self.num_layers):
            z, a = self.fully_connected_layer(a, self.sizes[i], '{}_layer{}'.format(self.scope, i))
            self.acts.append(a)
            self.zs.append(z)

        self.acct_mat = tf.equal(tf.argmax(a, 1), tf.argmax(self.labels, 1))
        self.acct_res = tf.reduce_sum(tf.cast(self.acct_mat, tf.float32))

        error = tf.subtract(a, self.labels)
        self.step = []
        for i in range(len(self.zs) - 1, -1, -1):
            error, weights_update, biases_update = self.backpropagation(error, self.zs[i], self.acts[i],
                '{}_layer{}'.format(self.scope, i + 1))
            self.step.append((weights_update, biases_update))

    def fully_connected_layer(self, input_act, output_dim, scope):
        with tf.variable_scope(scope):
            weights = tf.get_variable("weights", [input_act.shape[1], output_dim],
                initializer=tf.random_normal_initializer())
            biases = tf.get_variable("biases", [output_dim],
                initializer=tf.constant_initializer())
            z = tf.add(tf.matmul(input_act, weights), biases)
            return z, tf.sigmoid(z)

    def backpropagation(self, input_error, z, a, scope):
        with tf.variable_scope(scope, reuse=True):
            weights = tf.get_variable("weights")
            tf.summary.histogram("a", a)
            tf.summary.image("weights", tf.reshape(weights, (1, weights.shape[0], weights.shape[1], 1)))
            biases = tf.get_variable("biases")
            error = tf.multiply(input_error, sigmoid_prime(z))
            roc_cost_over_biases = error
            roc_cost_over_weights = tf.matmul(tf.transpose(a), error)
            output_error = tf.matmul(error, tf.transpose(weights))
            weights = tf.assign(weights, tf.subtract(weights, tf.multiply(self.learning_rate, roc_cost_over_weights)))
            biases = tf.assign(biases, tf.subtract(biases, tf.multiply(self.learning_rate, tf.reduce_mean(roc_cost_over_biases,
                axis=[0]))))
            return output_error, weights, biases

    def load_data(self):
        #TODO Proponowałbym tu sprawne uzycie tf.data oraz jako osobny moduł parsowanie mnista.
        pass

    def train(self, batch_size=10, batch_num=10000):
        #TODO generlize it to other datasets using tf.data for example
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("./demo/{}_{}".format(self.scope, self.run_number), sess.graph)
            sess.run(tf.global_variables_initializer())
            for i in range(batch_num):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(self.step, feed_dict={self.features: batch_xs, self.labels: batch_ys})
                if i % 1000 == 0:
                    merged = tf.summary.merge_all()
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, res = sess.run([merged, self.acct_res], options=run_options, run_metadata=run_metadata,
                        feed_dict={self.features: mnist.test.images[:1000], self.labels: mnist.test.labels[:1000]})
                    writer.add_summary(summary, i)
                    writer.add_run_metadata(run_metadata, 'step%d' % i)
                    print("{}%".format(res / 10))
            res = sess.run(self.acct_res, feed_dict={self.features: mnist.test.images, self.labels: mnist.test.labels})
            print("{}%".format(res / len(mnist.test.labels) * 100))
            writer.close()
            #TODO save model

    def infer(self, x):
        #TODO restore model
        with tf.Session() as sess:
            res = sess.run(self.activations[-1], feed_dict={self.features: x})
        return res

    def test(self, x, y):
        #TODO restore model
        with tf.Session() as sess:
            res = sess.run(self.acct_res, feed_dict={self.features: x, self.labels: y})
        return res

if __name__ == '__main__':
    if not os.path.isfile(file_name):
        with open(file_name, 'w+') as file:
            file.write(str(0))

    NN = NeuralNetwork([784, 50, 30, 10], scope='BP')
    NN.build()
    NN.train()
