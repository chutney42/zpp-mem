import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from utils import *

from tensorflow.keras.utils import to_categorical
(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.mnist.load_data()

train_features = train_features.reshape(train_features.shape[0], -1)
test_features = test_features.reshape(test_features.shape[0], -1)
test_labels = to_categorical(test_labels)
train_labels = to_categorical(train_labels)

train_features = train_features.astype('float32') / 255.0
test_features = test_features.astype('float32') / 255.0


def shuffle(features, labels):
    tmp = list(zip(features, labels))
    np.random.shuffle(tmp)
    return list(zip(*tmp))


class NeuralNetwork(object):
    def __init__(self, sizes, learning_rate=0.5, scope="main"):
        self.scope = scope
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.learning_rate = tf.constant(learning_rate)
        self.labels = tf.placeholder(tf.float32, [None, self.sizes[-1]])
        self.features = tf.placeholder(tf.float32, [None, self.sizes[0]])

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

    def train(self, batch_size=10, epoch=150):
        #TODO generlize it to other datasets using tf.data for example
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("./demo/{}".format(self.scope))
            writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())
            for e in range(epoch):
                features, labels = shuffle(train_features, train_labels)
                for i in np.arange(0, len(features), batch_size):
                    batch_xs = features[i: i+batch_size]
                    batch_ys = labels[i: i+batch_size]
                    sess.run(self.step, feed_dict={self.features: batch_xs, self.labels: batch_ys})

                res = sess.run(self.acct_res, feed_dict={self.features: test_features[:1000], self.labels:
                    test_labels[:1000]})
                print("{}%".format(res / 10))

            res = sess.run(self.acct_res, feed_dict={self.features: test_features, self.labels: test_labels})
            print("{}%".format(res / len(test_labels) * 100))
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
    NN = NeuralNetwork([784, 50, 30, 10], scope='BP')
    NN.build()
    NN.train()
