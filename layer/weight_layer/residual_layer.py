import tensorflow as tf

from layer import Layer
from layer.weight_layer.convolutional_layers import ConvolutionalLayer


class ResidualLayer(Layer):

    def __init__(self, sequence, trainable=True, scope="residual_layer"):
        super().__init__(trainable, scope=scope)
        self.propagator = None
        self.sequence = sequence
        self.shortcut_conv = None
        self.conv_func = None

    def __str__(self):
        s = f"ResidualLayer["
        for layer in self.sequence:
            s = s + f", {str(layer)}"
        s = s + "]"
        return s

    def build_forward(self, input, remember_input=False, gather_stats=False):
        with tf.variable_scope(self.scope, tf.AUTO_REUSE):
            if remember_input:
                self.input = input

            for i, layer in enumerate(self.sequence):
                layer.scope = f"{self.scope}_{i}_{layer.scope}"

            residual = input
            for layer in self.sequence:
                residual = layer.build_forward(residual, remember_input=True, gather_stats=gather_stats)

            res_shape = residual.shape
            input_shape = input.shape
            stride_width = int(round(input_shape[1].value / res_shape[1].value))
            stride_height = int(round(input_shape[2].value / res_shape[2].value))
            equal_channels = input_shape[3].value == res_shape[3].value

            if stride_width > 1 or stride_height > 1 or not equal_channels:
                self.shortcut_conv = ConvolutionalLayer(num_of_filters=res_shape[3],
                                  filter_dim=(1, 1),
                                  strides=[1, stride_width, stride_height, 1],
                                  padding="VALID",
                                  scope=f"{self.scope}_{len(self.sequence)}_shortcut_convolution")
                self.shortcut_conv.func = self.conv_func
                shortcut = self.shortcut_conv.build_forward(input)
            else:
                shortcut = input

            output = tf.add(shortcut, residual)
            if gather_stats:
                tf.summary.histogram("input", input, family=self.scope)
                tf.summary.histogram("output", output, family=self.scope)

            return output

    def gather_stats_backward(self, gradients):
        tf.summary.histogram("delta_input", gradients[0], family=self.scope)

