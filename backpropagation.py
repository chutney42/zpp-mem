import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from utils import *
from layer import *
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class NeuralNetwork(object):
    def __init__(self, input_dim, sequence, output_dim, learning_rate=0.1, scope="main"):
        self.scope = scope
        self.sequence = sequence
        self.learning_rate = tf.constant(learning_rate)

        self.features = tf.placeholder(tf.float32, [None, input_dim])
        self.labels = tf.placeholder(tf.float32, [None, output_dim])

    def build(self):
        a = self.build_forward()
        self.build_test(a)
        self.build_backward(a)

    def build_forward(self):
        a = self.features
        for i, layer in enumerate(self.sequence):
            layer.scope = '{}_{}_{}'.format(self.scope, layer.scope, i)
            a = layer.build_forward(a)
        return a

    def build_test(self, a):
        self.acct_mat = tf.equal(tf.argmax(a, 1), tf.argmax(self.labels, 1))
        self.acct_res = tf.reduce_sum(tf.cast(self.acct_mat, tf.float32))

    def build_backward(self, output_vec):
        error = tf.subtract(output_vec, self.labels)
        self.step = []
        for i, layer in reversed(list(enumerate(self.sequence))):
            error = layer.build_backward(error)
            if (layer.trainable):
                self.step.append(layer.step)

    def load_data(self):
        #TODO Proponowałbym tu sprawne uzycie tf.data oraz jako osobny moduł parsowanie mnista.
        pass

    def train(self, batch_size=10, batch_num=100000):
        #TODO generlize it to other datasets using tf.data for example
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("./demo/{}".format(self.scope))
            writer.add_graph(sess.graph)
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
    #NN = NeuralNetwork([784, 50, 30, 10], scope='BP')
    NN = NeuralNetwork(784,
                       [FullyConnected(50),
                        Sigmoid(),
                        FullyConnected(30),
                        Sigmoid(),
                        FullyConnected(10),
                        Sigmoid()],
                       10,
                       'BP')
    NN.build()
    NN.train()
