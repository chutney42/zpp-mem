from layer.weight_layer.convolutional_layers import ConvolutionalLayer, WeightLayer
import tensorflow as tf

class ResidualLayer(WeightLayer):

    def __init__(self, sequenceA, learning_rate=0.5, momentum=0.0, scope="residual_layer"):
        super().__init__(learning_rate, momentum, scope=scope)
        self.sequenceA = sequenceA

    def build_forward(self, input_vec, remember_input=True, gather_stats=True):
        residual = input_vec
        for block in self.sequenceA:
            residual = block.head.build_forward(residual, remember_input=True)
            for layer in block.tail:
                residual = layer.build_forward(residual, remember_input=True)

        res_shape = residual.shape
        input_shape = input_vec.shape

        stride_width = int(round(input_shape[1] / res_shape[1]))
        stride_height = int(round(input_shape[2] / res_shape[2]))
        equal_channels = input_shape[3] == input_shape[3]
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            self.shortcut_conv = ConvolutionalLayer(number_of_filters=res_shape[3],
                              filter_dim=(1, 1),
                              stride=[1, stride_width, stride_height, 1],
                              padding="VALID")
            self.shortcut_conv.propagator = self.blockA.head.propagator
            residual = self.shortcut_conv.build_forward(residual)
        else:
            self.shortcut_conv = None
        return input_vec + residual

    def build_backward(self, error, gather_stats=True):
        propagated_error = self.build_propagate(error, gather_stats)

        self.step = []
        if self.shortcut_conv is not None:
            error = self.step.append(self.shortcut_conv.buid_backward(error))
        for block in reversed(self.sequenceA):
            for layer in reversed(list(block)):
                error = layer.build_backward(error)
                if layer.trainable:
                    self.step.append(layer.step)

        return error + tf.gradients(in, )
