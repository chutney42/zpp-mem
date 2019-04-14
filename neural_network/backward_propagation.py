import tensorflow as tf
from neural_network.neural_network import NeuralNetwork


class BackwardPropagation(NeuralNetwork):
    def build_forward(self):
        a = self.features
        for block in self.sequence:
            a = block.head.build_forward(a, remember_input=True, gather_stats=self.gather_stats)
            for layer in block.tail:
                a = layer.build_forward(a, remember_input=True, gather_stats=self.gather_stats)
        return a

    def build_backward(self, error):
        self.step = []
        for block in reversed(self.sequence):
            for layer in reversed(block):
                error = layer.build_backward(error, gather_stats=self.gather_stats)
                if layer.trainable:
                    self.step.append(layer.step)
