import tensorflow as tf
from layer import *
from loader import *
from neuralnetwork import NeuralNetwork
from propagate import backpropagation


class BPNeuralNetwork(NeuralNetwork):
    def __init__(self, input_dim, sequence, *args, **kwargs):
        for block in sequence:
            block.head.propagate = backpropagation
        super().__init__(input_dim, sequence, *args, **kwargs)

    def build(self):
        self.result = self.__build_forward()
        self.__build_test(self.result)
        self.__build_backward(self.result)

    def __build_forward(self):
        a = self.features
        for block in self.sequence:
            a = block.head.build_forward(a)
            for layer in block.tail:
                a = layer.build_forward(a)
        return a

    def __build_test(self, a):
        self.acct_mat = tf.equal(tf.argmax(a, 1), tf.argmax(self.labels, 1))
        self.acct_res = tf.reduce_sum(tf.cast(self.acct_mat, tf.float32))
        if self.gather_stats:
            tf.summary.scalar("result", self.acct_res)

    def __build_backward(self, output_vec):
        error = tf.subtract(output_vec, self.labels)
        self.step = []
        for block in reversed(self.sequence):
            for layer in reversed(list(block)):
                error = layer.build_backward(error)
                if layer.trainable:
                    self.step.append(layer.step)


if __name__ == '__main__':
    training, test = load_mnist()
    NN = BPNeuralNetwork(784,
                         [Block([FullyConnected(50), BatchNormalization(), Sigmoid()]),
                          Block([FullyConnected(30), BatchNormalization(), Sigmoid()]),
                          Block([FullyConnected(10), Sigmoid()])],
                         10,
                         0.1,
                         'BP')
    NN.train(training, test)
