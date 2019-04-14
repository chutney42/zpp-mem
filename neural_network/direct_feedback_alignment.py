import tensorflow as tf
from neural_network.neural_network import NeuralNetwork
from propagator.direct_propagator import DirectFixedRandom


class DirectFeedbackAlignment(NeuralNetwork):
    def __init__(self, types, shapes, sequence, cost_function_name,
                 propagator_initializer=tf.random_normal_initializer(), *args, **kwargs):
        propagator = DirectFixedRandom(shapes[1][1].value, propagator_initializer)
        super().__init__(types, shapes, sequence, cost_function_name, propagator, *args, **kwargs)

    def build_forward(self):
        a = self.features
        for block in self.sequence:
            for layer in block:
                a = layer.build_forward(a, remember_input=False, gather_stats=self.gather_stats)
        return a

    def build_backward(self, output_error):
        self.step = []
        a = self.features
        for i, block in enumerate(self.sequence):
            for layer in block:
                a = layer.build_forward(a, remember_input=True, gather_stats=self.gather_stats)
            if i + 1 < len(self.sequence):
                error = self.sequence[i + 1].head.build_propagate(output_error, gather_stats=self.gather_stats)
            else:
                error = output_error
            for layer in reversed(block.tail):
                error = layer.build_backward(error, gather_stats=self.gather_stats)
                if layer.trainable:
                    self.step.append(layer.step)
            block.head.build_update(error, gather_stats=self.gather_stats)
            self.step.append(block.head.step)
