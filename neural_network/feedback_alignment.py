import tensorflow as tf

from neural_network.backward_propagation import BackwardPropagation
from layer.weight_layer.convolutional_layers import ConvolutionalLayer
from layer.weight_layer.fully_connected import FullyConnected
from custom_operations import feedback_alignment_fc, feedback_alignment_conv


class FeedbackAlignment(BackwardPropagation):
    def __init__(self, types, shapes, sequence, *args, **kwargs):
        for layer in sequence:
            if isinstance(layer, ConvolutionalLayer):
                layer.func = feedback_alignment_conv
            elif isinstance(layer, FullyConnected):
                layer.func = feedback_alignment_fc
        super().__init__(types, shapes, sequence, *args, **kwargs)
        
