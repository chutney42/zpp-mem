import tensorflow as tf

from layer.layer import Layer


class Pool(Layer):
    def __init__(self, pooling_function, kernel_size, strides, padding="VALID", data_format='NHWC', scope="pool_layer"):
        super().__init__(trainable=False, scope=scope)
        self.kernel_size = [1] + kernel_size + [1]
        self.strides = [1] + strides + [1]
        if len(self.kernel_size) > 4:
            self.kernel_size = kernel_size
        if len(self.strides) > 4:
            self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.pooling_function = pooling_function
        self.pooling_function_name = None

    def __str__(self):
        return f"{self.pooling_function_name}({self.kernel_size}, {self.strides}, {self.padding})"

    def build_forward(self, input_vec, remember_input=True, gather_stats=True):
        if remember_input:
            self.input_vec = input_vec
        with tf.variable_scope(self.scope, tf.AUTO_REUSE):
            output = self.pooling_function(input_vec, self.kernel_size, self.strides, self.padding, self.data_format)
            return output

    def build_backward(self, error, gather_stats=True):
        with tf.variable_scope(self.scope):
            pre_pool = self.restore_input()
            post_pool = self.pooling_function(pre_pool, self.kernel_size, self.strides, self.padding, self.data_format)
            backprop_error = tf.gradients(post_pool, pre_pool, error)
            return backprop_error[0]


class MaxPool(Pool):
    def __init__(self, kernel_size, strides, padding="VALID", data_format='NHWC', scope="max_pool_layer"):
        super().__init__(tf.nn.max_pool, kernel_size, strides, padding, data_format, scope)
        self.pooling_function_name = "MaxPool"


class AveragePool(Pool):
    def __init__(self, kernel_size, strides, padding="VALID", data_format='NHWC', scope="avg_pool_layer"):
        super().__init__(tf.nn.avg_pool, kernel_size, strides, padding, data_format, scope)
        self.pooling_function_name = "AveragePool"
