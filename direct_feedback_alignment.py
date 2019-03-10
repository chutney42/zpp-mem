import tensorflow as tf
from layer import *
from loader import *
from neuralnetwork import NeuralNetwork
from propagate import direct_feedback_alignment


class DirectFeedbackAlignment(NeuralNetwork):
    def set_propagate_functions(self):
        for block in self.sequence:
            block.head.propagate_func = direct_feedback_alignment

    def build_forward(self):
        a = self.features
        for block in self.sequence:
            for layer in block:
                a = layer.build_forward(a, remember_input=False)
        return a

    def build_backward(self, output_vec):
        output_error = tf.subtract(output_vec, self.labels)
        self.step = []
        a = self.features
        for i, block in enumerate(self.sequence):
            for layer in block:
                a = layer.build_forward(a, remember_input=True)
            if i + 1 < len(self.sequence):
                error = self.sequence[i + 1].head.build_propagate(output_error)
            else:
                error = output_error
            for layer in reversed(block.tail):
                error = layer.build_backward(error)
                if layer.trainable:
                    self.step.append(layer.step)
            block.head.build_update(error)
            self.step.append(block.head.step)

if __name__ == '__main__':
    training, test = load_mnist()
    NN = DirectFeedbackAlignment(784,
                                 [Block([FullyConnected(50), BatchNormalization(), Sigmoid()]),
                                  Block([FullyConnected(30), BatchNormalization(), Sigmoid()]),
                                  Block([FullyConnected(10), Sigmoid()])],
                                 10,
                                 0.1,
                                 'DFA')
    NN.train(training, test)
