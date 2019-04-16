import tensorflow as tf
from neural_network.neural_network import NeuralNetwork


class BackwardPropagation(NeuralNetwork):
    def build_forward(self):
        a = self.features
        for layer in self.sequence:
            a = layer.build_forward(a, remember_input=False, gather_stats=self.gather_stats)
        return a
