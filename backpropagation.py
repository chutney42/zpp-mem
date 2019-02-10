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
    def __init__(self, sizes2, eta=0.5, scope_name="main_scope"):
        self.num_layers = len(sizes2)
        self.types = [a[0] for a in sizes2]
        self.sizes = [a[1] for a in sizes2]
        self.y = tf.placeholder(tf.float32, [None, self.sizes[-1]])
        self.eta = tf.constant(eta)
        self.x = tf.placeholder(tf.float32, [None, self.sizes[0]])
        self.vars = [self.x]

        def normalize(i):
            return self.types[i] == "n"

        def activate(i):
            return self.types[i] == "a"

        def fully_connect(i):
            return self.types[i] == "f"

        a = self.x
        # FORWARD PASS
        for i in range(1, self.num_layers):
            if fully_connect(i):
                self.vars.append(a)
                a = self.fully_connected_layer(a, self.sizes[i], '{}_layer{}'.format(scope_name, i))
            elif normalize(i):
                a_prev = a
                a, mean, var = self.normalization_layer(a, self.sizes[i], '{}_layer{}'.format(scope_name, i))
                self.vars.append((a_prev, mean, var))
            elif activate(i):
                self.vars.append((a, 13))
                a = self.activate_layer(a, tf.sigmoid, '{}_layer{}'.format(scope_name, i))
        output = a
        da = tf.subtract(output, self.y)
        self.step = []

        # BACKWARD PASS
        for i in range(len(self.vars) - 1, -1, -1):
            if normalize(i):
                a, mean, var = self.vars[i]
                da, gamma_update, beta_update = self.backprogation_normalization(da, a, self.sizes[i], eta, mean, var,
                                                                                 '{}_layer{}'.format(scope_name, i))
                self.step.append((beta_update, gamma_update))

            elif fully_connect(i):
                a = self.vars[i]
                da, w_update, b_update = self.backpropagation_fully_connected(da, a, eta,
                                                                              '{}_layer{}'.format(scope_name, i))
                self.step.append((w_update, b_update))

            elif activate(i):
                a, _ = self.vars[i]

                da = self.backpropagation_activate(da, a, '{}_layer{}'.format(scope_name, i))
        self.acct_mat = tf.equal(tf.argmax(output, 1), tf.argmax(self.y, 1))
        self.acct_res = tf.reduce_sum(tf.cast(self.acct_mat, tf.float32))

    def fully_connected_layer(self, x, y_dim, scope):
        with tf.variable_scope(scope):
            w = tf.get_variable("weights", [x.shape[1], y_dim], initializer=tf.random_normal_initializer())

            b = tf.get_variable("biases", [y_dim], initializer=tf.constant_initializer())
            tf.summary.histogram("w:", w)
            tf.summary.image("w_image", tf.reshape(w, (1, w.shape[0], w.shape[1], 1)))
            tf.summary.histogram("b:", b)

            z = tf.add(tf.matmul(x, w), b)
            return z

    def activate_layer(self, z, act_func, scope):
        with tf.variable_scope(scope):
            a = act_func(z)
            return a

    def backpropagation_fully_connected(self, da, a, eta, scope):
        with tf.variable_scope(scope, reuse=True):
            w = tf.get_variable("weights")
            b = tf.get_variable("biases")
            db = da
            dw = tf.matmul(tf.transpose(a), da)
            dal = tf.matmul(da, tf.transpose(w))  # da lower
            return dal, \
                   tf.assign(w, tf.subtract(w, tf.scalar_mul(eta, dw))), \
                   tf.assign(b, tf.subtract(b, tf.scalar_mul(eta, tf.reduce_mean(db, axis=[0]))))

    def backpropagation_activate(self, da, z, scope):
        with tf.variable_scope(scope, reuse=True):
            dz = tf.multiply(da, sigmoid_prime(z))
            return dz

    def backprogation_normalization(self, da, z, y_dim, eta, batch_mean, batch_var, scope):
        with tf.variable_scope(scope, reuse=True):
            epsilon = 0.001  # In case batch_var=0, division by zero is not cool
            gamma = tf.get_variable("gamma")
            beta = tf.get_variable("beta")
            N = y_dim

            z_normalized = (z - batch_mean) / tf.sqrt(batch_var + epsilon)

            z_mu = z - batch_mean
            std_inv = 1 / tf.sqrt(tf.add(batch_var, epsilon))
            dz_norm = da * gamma
            dvar = tf.matmul(tf.scalar_mul(-.5, tf.reduce_sum(tf.matmul(dz_norm, z_mu), 0)),   tf.pow(std_inv, 3))
            dmu = tf.add(tf.reduce_sum(tf.matmul(dz_norm, -std_inv), 0),
                         tf.matmul(dvar, tf.reduce_mean(tf.scalar_mul(-2., z_mu), 0)))

            dz = tf.add(tf.add(tf.matmul(dz_norm, std_inv), tf.matmul(tf.scalar_mul(2. / N, dvar), z_mu)),
                        tf.scalar_mul(1. / N, dmu))

            dgamma = tf.reduce_sum(tf.matmul(da, z_normalized), 0)
            dbeta = tf.reduce_sum(da, 0)

            return dz, \
                   tf.assign(gamma, tf.subtract(gamma, tf.multiply(eta, dgamma))), \
                   tf.assign(beta, tf.subtract(dbeta, tf.multiply(eta, dbeta)))

    def normalization_layer(self, x, y_dim, scope):
        with tf.variable_scope(scope):
            epsilon = 0.001  # In case batch_var=0, division by zero is not cool
            gamma = tf.get_variable("gamma", [y_dim], initializer=tf.ones_initializer())
            beta = tf.get_variable("beta", [y_dim], initializer=tf.ones_initializer())
            z = x

            batch_mean, batch_var = tf.nn.moments(z, [0])
            z_normalized = (z - batch_mean) / tf.sqrt(batch_var + epsilon)

            z_normalized = gamma * z_normalized + beta
            return z_normalized, batch_mean, batch_var

    def load_data(self):
        pass

    def train(self, batch_size=10, batch_num=100000):
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("logs/miron", sess.graph)
            sess.run(tf.global_variables_initializer())
            for i in range(batch_num):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(self.step, feed_dict={self.x: batch_xs, self.y: batch_ys})
                if i % 1000 == 0:
                    merged = tf.summary.merge_all()
                    summary, res = sess.run([merged, self.acct_res], feed_dict={self.x: mnist.test.images[:1000],
                    writer.add_summary(summary, i)
            # TODO save model
            writer.close()

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
    NN = NeuralNetwork([("-", 784), ("f", 50), ("a", 50),("a",50), ("a", 50), ("f", 10), ("a", 10)])
    # NN = NeuralNetwork([("-", 784), ("f", 50), ("n", 50),("a", 50), ("f", 10), ("a", 10)])
    NN.load_data()
    NN.train()
