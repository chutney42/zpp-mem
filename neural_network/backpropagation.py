from neural_network.backward_propagation import BackwardPropagation
from layer.weight_layer.convolutional_layers import ConvolutionalLayer
from layer.weight_layer.fully_connected import FullyConnected
from layer.weight_layer.weight_layer import WeightLayer


class Backpropagation(BackwardPropagation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer in self.sequence:
            if isinstance(layer, ConvolutionalLayer):
                layer.func = tf.nn.conf2d
            elif isinstance(layer, FullyConnected):
                layer.func = lambda x, w : tf.matmul(x, w)
