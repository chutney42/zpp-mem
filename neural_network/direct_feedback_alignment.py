import tensorflow as tf
from functools import partial
from neural_network.neural_network import NeuralNetwork
from neural_network.backward_propagation import BackwardPropagation
from layer.weight_layer import WeightLayer, ConvolutionalLayer, FullyConnected
from custom_operations import direct_feedback_alignment_fc, direct_feedback_alignment_conv


class DirectFeedbackAlignment(BackwardPropagation):

    def __init__(self, types, shapes, sequence, *args, **kwargs):
        self.error_container = []
        for layer in sequence:
            if isinstance(layer, ConvolutionalLayer):
                layer.func = partial(direct_feedback_alignment_conv,
                                     output_dim=shapes[1][1].value,
                                     error_container=self.error_container)
            elif isinstance(layer, FullyConnected):
                layer.func = partial(direct_feedback_alignment_fc,
                                     output_dim=shapes[1][1].value,
                                     error_container=self.error_container)
        super().__init__(types, shapes, sequence, *args, **kwargs)

    def build_error(self, cost, result):
        self.error_container.append(tf.gradients(cost, result, name="error")[0])
        return self.error_container[0]


class DirectFeedbackAlignmentMem(DirectFeedbackAlignment): # TODO

    def build_forward(self):
        with tf.name_scope("forward"):
            a = self.features
            for layer in self.sequence:
                a = layer.build_forward(a, remember_input=False, gather_stats=self.gather_stats)
            return a

    def build_backward(self, error, output):
        with tf.name_scope("backward"):

            def build_partial_backward(error, output, first, last, step):
                for layer in reversed(self.sequence[first + 1 : last]):
                    error, output = layer.build_backward(error, output, self.optimizer, self.gather_stats)
                    if layer.trainable:
                        step.append(layer.step)
                self.sequence[first].build_update(error, output, self.optimizer)
                layer = self.sequence[first]
                if layer.trainable:
                    step.append(layer.step)
                return step

            step = []
            i = 0
            a = self.sequence[0].build_forward(self.features, remember_input=True, gather_stats=self.gather_stats)

            for j, layer in enumerate(self.sequence[1:]):
                a = layer.build_forward(a, remember_input=True, gather_stats=self.gather_stats)
                if isinstance(layer, WeightLayer):
                    perror = layer.build_propagate(1.0, a)
                    step = build_partial_backward(perror, a, i, j + 1, step)
                    i = j + 1

            step = build_partial_backward(error, a, i, j + 1, step)

            return tf.group(step)
