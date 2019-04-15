import tensorflow as tf

from layer import ResidualLayer
from layer.weight_layer.convolutional_layers import ConvolutionalLayer
from layer.weight_layer.fully_connected import FullyConnected
from neural_network.backward_propagation import BackwardPropagation


class Backpropagation(BackwardPropagation):
    def __init__(self, types, shapes, sequence, *args, **kwargs):
        self._initialize_custom_gradients(sequence)
        super().__init__(types, shapes, sequence, *args, **kwargs)
        
    def _initialize_custom_gradients(self, sequence):
        for layer in sequence:
            if isinstance(layer, ConvolutionalLayer):
                layer.func = tf.nn.conv2d
            elif isinstance(layer, FullyConnected):
                layer.func = lambda x, w: tf.matmul(x, w)
            elif isinstance(layer, ResidualLayer):
                layer.conv_func = tf.nn.conv2d
                self._initialize_custom_gradients(layer.sequence)
