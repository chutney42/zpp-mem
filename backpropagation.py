import os
import tensorflow as tf
from utils import *
from layer import *
from loader import *
from neuralnetwork import NeuralNetwork


class BPNeuralNetwork(NeuralNetwork):
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
        tf.summary.scalar("result", self.acct_res)

    def build_backward(self, output_vec):
        error = tf.subtract(output_vec, self.labels)
        self.step = []
        for i, layer in reversed(list(enumerate(self.sequence))):
            error = layer.build_backward(error)
            if layer.trainable:
                self.step.append(layer.step)


if __name__ == '__main__':
    training, test = load_mnist()
    NN = BPNeuralNetwork(784,
                       [FullyConnected(50),
                        BatchNormalization(),
                        Sigmoid(),
                        FullyConnected(30),
                        BatchNormalization(),
                        Sigmoid(),
                        FullyConnected(10),
                        Sigmoid()],
                       10,
                       0.1,
                       'BP')
    NN.train(training, test)
