from layer import *
from loader import *
from backward_propagation import BackwardPropagation
from propagate import backpropagation


class Backpropagation(BackwardPropagation):
    def set_propagate_functions(self):
        for block in self.sequence:
            block.head.propagate_func = backpropagation


if __name__ == '__main__':
    training, test = load_mnist()
    NN = Backpropagation(784,
                         [Block([FullyConnected(50), BatchNormalization(), Sigmoid()]),
                          Block([FullyConnected(30), BatchNormalization(), Sigmoid()]),
                          Block([FullyConnected(10), Sigmoid()])],
                         10,
                         0.1,
                         'BP')
    NN.train(training, test)
