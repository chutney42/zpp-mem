import tensorflow as tf
from functools import partial

from layer import ResidualLayer
from neural_network.neural_network import NeuralNetwork
from neural_network.backward_propagation import BackwardPropagation
from layer.weight_layer.convolutional_layers import ConvolutionalLayer
from layer.weight_layer.fully_connected import FullyConnected
from custom_operations import direct_feedback_alignment_fc, direct_feedback_alignment_conv


class DirectFeedbackAlignment(BackwardPropagation):
    def __init__(self, types, shapes, sequence, *args, **kwargs):
        self.error_container = []
        self._initialize_custom_gradients(sequence, shapes)
        super().__init__(types, shapes, sequence, *args, **kwargs)

    def _initialize_custom_gradients(self, sequence, shapes):
        for layer in sequence:
            if isinstance(layer, ConvolutionalLayer):
                layer.func = partial(direct_feedback_alignment_conv,
                                     output_dim=shapes[1][0].value,
                                     error_container=self.error_container)
            elif isinstance(layer, FullyConnected):
                layer.func = partial(direct_feedback_alignment_fc,
                                     output_dim=shapes[1][0].value,
                                     error_container=self.error_container)
            elif isinstance(layer, ResidualLayer):
                layer.conv_func = partial(direct_feedback_alignment_conv,
                                     output_dim=shapes[1][0].value,
                                     error_container=self.error_container)

                self._initialize_custom_gradients(layer.sequence, shapes)

     def build(self):
            self.result = self.build_forward()
            self.cost = self.cost_function(self.labels, self.result)
            self.build_test(self.result)
            self.error_container.append(tf.gradients(self.cost, self.result, name="error")[0])
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.step = self.optimizer.minimize(self.cost)
            self.step = tf.group([self.step, update_ops])



class DirectFeedbackAlignmentMem(NeuralNetwork): # TODO
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
