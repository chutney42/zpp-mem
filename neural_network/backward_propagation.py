import tensorflow as tf
from neural_network.neural_network import NeuralNetwork


class BackwardPropagation(NeuralNetwork):
    def build_forward(self):
        with tf.name_scope("forward"):
            a = self.features
            for layer in self.sequence:
                a = layer.build_forward(a, remember_input=True, gather_stats=self.gather_stats)
            return a

    def build_backward(self, error, output):
        with tf.name_scope("backward"):
            step = []
            for layer in reversed(self.sequence):
                error, output = layer.build_backward(error, output, self.optimizer, self.gather_stats)
                if layer.trainable:
                    step.append(layer.step)
            return tf.group(step)
