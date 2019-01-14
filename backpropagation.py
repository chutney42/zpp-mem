import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def sigmoid_prime(x):
    return tf.multiply(tf.sigmoid(x), (tf.constant(1.0) - tf.sigmoid(x)))


class NeuralNetwork(object):
    def __init__(self, sizes, eta=0.5, scope_name="main_scope"):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.y = tf.placeholder(tf.float32, [None, self.sizes[-1]])
        self.eta = tf.constant(eta)
        self.x = tf.placeholder(tf.float32, [None, self.sizes[0]])
        self.acts = [self.x]
        self.zs = []
        a = self.x
        for i in range(1, self.num_layers):
            z, a = self.fully_connected_layer(a, self.sizes[i], tf.sigmoid, '{}_h{}'.format(scope_name, i))
            self.acts.append(a)
            self.zs.append(z)
        da = tf.subtract(a, self.y)
        self.step = []
        for i in range(len(self.zs) - 1, -1, -1):
            da, w_update, b_update = self.backpropagation(da, self.zs[i], self.acts[i], self.eta,
                                                          '{}_h{}'.format(scope_name, i + 1))
            self.step.append((w_update, b_update))
        self.acct_mat = tf.equal(tf.argmax(self.acts[-1], 1), tf.argmax(self.y, 1))
        self.acct_res = tf.reduce_sum(tf.cast(self.acct_mat, tf.float32))

    def fully_connected_layer(self, x, y_dim, act_func, scope):
        with tf.variable_scope(scope):
            w = tf.get_variable("weights", [x.shape[1], y_dim], initializer=tf.random_normal_initializer())
            b = tf.get_variable("biases", [y_dim], initializer=tf.constant_initializer())
            z = tf.add(tf.matmul(x, w), b)
            return z, act_func(z)

    def backpropagation(self, da, z, a, eta, scope):
        with tf.variable_scope(scope, reuse=True):
            w = tf.get_variable("weights")
            b = tf.get_variable("biases")
            dz = tf.multiply(da, sigmoid_prime(z))
            db = dz
            dw = tf.matmul(tf.transpose(a), dz)
            dal = tf.matmul(dz, tf.transpose(w)) #da lower
            return dal, tf.assign(w, tf.subtract(w, tf.multiply(eta, dw))), tf.assign(b, tf.subtract(b, tf.multiply(eta, tf.reduce_mean(db, axis=[0]))))

    def load_data(self):
        # TODO Proponowałbym tu sprawne uzycie tf.data oraz jako osobny moduł parsowanie mnista.
        pass

    def train(self, batch_size=10, batch_num=10000):
        # TODO generlize it to other datasets using tf.data for example
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(batch_num):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(self.step, feed_dict={self.x: batch_xs, self.y: batch_ys})
                if i % 1000 == 0:
                    res = sess.run(self.acct_res, feed_dict={self.x: mnist.test.images[:1000],
                                                             self.y: mnist.test.labels[:1000]})
                    print(res)
            # TODO save model

    def infer(self, x):
        # TODO restore model
        with tf.Session() as sess:
            res = sess.run(self.acts[-1], feed_dict={self.x: x})
        return res

    def test(self, x, y):
        # TODO restore model
        with tf.Session() as sess:
            res = sess.run(self.acct_res, feed_dict={self.x: x, self.y: y})
        return res


if __name__ == '__main__':
    NN = NeuralNetwork([784, 50, 50, 50, 30, 10])
    NN.load_data()
    NN.train()
