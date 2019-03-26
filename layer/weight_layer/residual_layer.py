from layer.layer import Layer
from layer.weight_layer.convolutional_layers import ConvolutionalLayer
from layer.weight_layer.weight_layer import WeightLayer


class ResidualLayer(Layer):

    def __init__(self, sequenceA, trainable=True,  scope="residual_layer"):
        super().__init__(trainable, scope=scope)
        self.propagator = None
        self.sequenceA = sequenceA

    def __str__(self):
        s = f"ResidualLayer["
        for layer in self.sequenceA:
            s = s + f", {str(layer)}"
        s = s + "]"
        return s

    def build_forward(self, input_vec, remember_input=False, gather_stats=True):
        for i, block in enumerate(self.sequenceA):
            for j, layer in enumerate(block):
                if isinstance(layer, WeightLayer):
                    layer.propagator = self.propagator
                layer.scope = f"{self.scope}_{layer.scope}_{i}_{j}"


        residual = input_vec
        for block in self.sequenceA:
            residual = block.head.build_forward(residual, remember_input=True)
            for layer in block.tail:
                residual = layer.build_forward(residual, remember_input=True)

        if remember_input:
            self.input_vec = input_vec

        res_shape = residual.shape
        input_shape = input_vec.shape

        stride_width = int(round(input_shape[1].value / res_shape[1].value))
        stride_height = int(round(input_shape[2].value / res_shape[2].value))
        print(stride_width)
        print(input_shape)
        print(res_shape)
        equal_channels = input_shape[3].value == res_shape[3].value
        print(equal_channels)
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            print("AGA")
            self.shortcut_conv = ConvolutionalLayer(number_of_filters=res_shape[3],
                              filter_dim=(1, 1),
                              stride=[stride_width, stride_height],
                              padding="VALID", scope=f"{self.scope}_convoluted_shortcut")
            self.shortcut_conv.propagator = self.propagator
            input_vec = self.shortcut_conv.build_forward(input_vec)
        else:
            self.shortcut_conv = None

        return input_vec + residual

    def build_backward(self, error, gather_stats=True):
        input_err = 1
        self.step = []
        if self.shortcut_conv is not None:
            input_err = self.shortcut_conv.build_backward(error)
            self.step.append(self.shortcut_conv.step)
        for block in reversed(self.sequenceA):
            for layer in reversed(list(block)):
                error = layer.build_backward(error)
                if layer.trainable:
                    self.step.append(layer.step)

        return error + input_err
