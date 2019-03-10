import tensorflow as tf
from layer import *
from loader import *
from backpropagation import BPNeuralNetwork
from propagate import direct_feedback_alignment


class DFANeuralNetwork(BPNeuralNetwork):
    def __init__(self, input_dim, sequence, *args, **kwargs):
        for block in sequence:
            block.head.propagate = direct_feedback_alignment
        super().__init__(input_dim, sequence, *args, **kwargs)

    def __build_forward(self):
        a = self.features
        for block in self.sequence:
            for layer in block:
                a = layer.build_forward(a, remember_input=False)
        return a

    def __build_backward(self, output_vec):
        error = tf.subtract(output_vec, self.labels)
        a = self.features
        for i, block in enumerate(self.sequence):
            for layer in block:
                a = layer.build_forward(a, remember_input=True)
            if i < len(self.sequence):
                error = block.head.build_propagate(error)
            for layer in reversed(block.tail):
                error = layer.build_backward(error)
                if layer.trainable:
                    self.step.append(layer.step)
            block.head.build_update(error)
            self.step.append(block.head.step)

if __name__ == '__main__':
    training, test = load_mnist()
    NN = DFANeuralNetwork(784,
                         [Block([FullyConnected(50), BatchNormalization(), Sigmoid()]),
                          Block([FullyConnected(30), BatchNormalization(), Sigmoid()]),
                          Block([FullyConnected(10), Sigmoid()])],
                         10,
                         0.1,
                         'DFA')
    NN.train(training, test)
