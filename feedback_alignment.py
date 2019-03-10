import tensorflow as tf
from layer import *
from loader import *
from backpropagation import BPNeuralNetwork
from propagate import feedback_alignment

class FANeuralNetwork(BPNeuralNetwork):
    def __init__(self, input_dim, sequence, *args, **kwargs):
        for block in sequence:
            block.head.propagate = feedback_alignment
        super().__init__(input_dim, sequence, *args, **kwargs)


if __name__ == '__main__':
    training, test = load_mnist()
    NN = FANeuralNetwork(784,
                         [Block([FullyConnected(50), BatchNormalization(), Sigmoid()]),
                          Block([FullyConnected(30), BatchNormalization(), Sigmoid()]),
                          Block([FullyConnected(10), Sigmoid()])],
                         10,
                         0.1,
                         'FA')
    NN.train(training, test)
