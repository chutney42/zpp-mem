import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from utils import *
from layer import *
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

file_name = "run_auto_increment"

class NeuralNetwork(object):
    def __init__(self, input_dim, sequence, output_dim, learning_rate=0.1, scope="main"):
        self.scope = scope
        self.sequence = sequence
        self.learning_rate = tf.constant(learning_rate)

        self.features = tf.placeholder(tf.float32, [None, input_dim])
        self.labels = tf.placeholder(tf.float32, [None, output_dim])

        with open(file_name) as file:
            self.run_number = int(file.read())

        self.run_number += 1

        with open(file_name, 'w') as file:
            file.write(str(self.run_number))

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
        tf.summary.scalar("result", self.acct_res)

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

    NN = NeuralNetwork(784,
                       [FullyConnected(50),
                        BatchNormalization(),
                        Sigmoid(),
                        FullyConnected(30),
                        BatchNormalization(),
                        Sigmoid(),
                        FullyConnected(10),
                        Sigmoid()],
                       10,
                       'BP')
    NN.build()
    NN.train()
