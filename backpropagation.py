import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from utils import *
from layer import *
from loader import *


class NeuralNetwork(object):
    def __init__(self, input_dim, sequence, output_dim, learning_rate=0.1, scope="main"):
        self.scope = scope
        self.sequence = sequence
        self.learning_rate = tf.constant(learning_rate)
        self.features = tf.placeholder(tf.float32, [None, input_dim])
        self.labels = tf.placeholder(tf.float32, [None, output_dim])
        self.result = None

    def build(self):
        self.result = self.build_forward()
        self.build_test(self.result)
        self.build_backward(self.result)

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

    def train(self, training_set, validation_set, batch_size=10, epoch=2, eval_period=1000):
        training_set = training_set.shuffle(200).batch(batch_size)
        iterator = tf.data.Iterator.from_structure(training_set.output_types,
                                                     training_set.output_shapes)
        train_init = iterator.make_initializer(training_set)
        next_batch = iterator.get_next()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("./demo/{}".format(self.scope))
            writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())
            counter = 0
            for e in range(epoch):
                sess.run(train_init)
                while True:
                    try:
                        batch_xs, batch_ys = sess.run(next_batch)
                        sess.run(self.step, feed_dict={self.features: batch_xs, self.labels: batch_ys})

                        if eval_period > 0 and counter % eval_period is 0:
                            print("iter: {}, acc: {}%".format(counter, self.validate(validation_set.take(1000), sess)))

                        counter += 1
                    except tf.errors.OutOfRangeError:
                        break
                res = self.validate(validation_set, sess)
                print("epoch {}:  {}%".format(e, res))

            res = self.validate(validation_set, sess)
            print("total {}%".format(res))
            # TODO save model

    def validate(self, validation_set, sess, batch_size=10):
        total_res = 0
        counter = 0
        next_batch = validation_set.batch(batch_size).make_one_shot_iterator().get_next()
        while True:
            try:
                batch_xs, batch_ys = sess.run(next_batch)
                res = sess.run(self.acct_res, feed_dict={self.features: batch_xs, self.labels: batch_ys})
                total_res += res
                counter += batch_size
            except tf.errors.OutOfRangeError:
                break
        return total_res / counter * 100

    def infer(self, x):
        # TODO restore model
        with tf.Session() as sess:
            res = sess.run(self.result, feed_dict={self.features: x})
        return res

    def test(self, data_set, batch_size=10):
        # TODO restore model
        next_batch = data_set.batch(batch_size).make_one_shot_iterator().get_next()
        total_res = 0
        counter = 0
        with tf.Session() as sess:
            while True:
                try:
                    batch_xs, batch_ys = sess.run(next_batch)
                    res = sess.run(self.acct_res, feed_dict={self.features: batch_xs, self.labels: batch_ys})
                    total_res += res
                    counter += batch_size
                except tf.errors.OutOfRangeError:
                    break
        return total_res / counter * 100


if __name__ == '__main__':

    training, test = load_mnist()
    NN = NeuralNetwork(784,
                       [FullyConnected(128),
                        Sigmoid(),
                        FullyConnected(64),
                        Sigmoid(),
                        FullyConnected(10),
                        Sigmoid()],
                       10,
                       'BP')
    NN.build()
    NN.train(training, test)

