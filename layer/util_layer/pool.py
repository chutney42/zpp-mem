import tensorflow as tf

from layer.layer import Layer


class Pool(Layer):
    def __init__(self, pooling_function, kernel_size, strides, padding="VALID", data_format='NHWC', scope="pool_layer"):
        super().__init__(trainable=False, scope=scope)
        self.kernel_size = None
        if kernel_size is not None:
            self.kernel_size = [1] + kernel_size + [1]
            if len(self.kernel_size) > 4:
                self.kernel_size = kernel_size
        self.strides = [1] + strides + [1]
        if len(self.strides) > 4:
            self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.pooling_function = pooling_function
        self.pooling_function_name = None

    def __str__(self):
        return f"{self.pooling_function_name}({self.kernel_size}, {self.strides}, {self.padding})"

    def build_forward(self, input, remember_input=False, gather_stats=False):
        if remember_input:
            self.save_input(input)
        with tf.name_scope(self.scope):
            if self.kernel_size is None:
                self.kernel_size = [1] + list(map(lambda x: x.value, input.shape[1:-1])) + [1]
            output = self.pooling_function(input, self.kernel_size, self.strides, self.padding, self.data_format)
            return output

class MaxPool(Pool):
    def __init__(self, kernel_size, strides, padding="VALID", data_format='NHWC', scope="max_pool_layer"):
        super().__init__(tf.nn.max_pool, kernel_size, strides, padding, data_format, scope)
        self.pooling_function_name = "MaxPool"


class AveragePool(Pool):
    def __init__(self, kernel_size, strides, padding="VALID", data_format='NHWC', scope="avg_pool_layer"):
        super().__init__(tf.nn.avg_pool, kernel_size, strides, padding, data_format, scope)
        self.pooling_function_name = "AveragePool"
