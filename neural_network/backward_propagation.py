from util.loader import *
from neural_network.neural_network import NeuralNetwork


class BackwardPropagation(NeuralNetwork):
    def build_forward(self):
        a = self.features
        for block in self.sequence:
            a = block.head.build_forward(a, remember_input=True)
            for layer in block.tail:
                a = layer.build_forward(a, remember_input=True)
        return a

    def build_backward(self, output_vec):
        error = tf.subtract(output_vec, self.labels)
        self.step = []
        for block in reversed(self.sequence):
            for layer in reversed(list(block)):
                error = layer.build_backward(error)
                if layer.trainable:
                    self.step.append(layer.step)
