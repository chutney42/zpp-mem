import tensorflow as tf

from layer.layer import Layer


class Pool(Layer):
    def __init__(self, pooling_function, ksize, strides, padding="VALID", data_format='NHWC', scope="max_pool_layer"):
        super().__init__(trainable=False, scope=scope)
        self.ksize = [1] + ksize + [1]
        self.strides = [1] + strides + [1]
        if len(self.ksize) > 4:
            self.ksize = ksize
        if len(self.strides) > 4:
            self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.pooling_function = pooling_function
        self.pooling_function_name = None

    def __str__(self):
        return f"{self.pooling_function_name}({self.ksize}, {self.strides}, {self.padding})"

    def build_forward(self, input_vec, remember_input=True, gather_stats=True):
        if remember_input:
            self.input_vec = input_vec
        with tf.variable_scope(self.scope, tf.AUTO_REUSE):
            output = self.pooling_function(input_vec, self.ksize, self.strides, self.padding, self.data_format)
            return output

    def build_backward(self, error, gather_stats=True):
        input_vec = self.restore_input()
        with tf.variable_scope(self.scope):
            xs = input_vec
            ys = tf.nn.max_pool(xs, self.ksize, self.strides, self.padding, self.data_format)
            backprop_error = tf.gradients(ys, xs, error)
            return backprop_error[0]


class MaxPool(Pool):
    def __init__(self, *args):
        super().__init__(tf.nn.max_pool, *args)
        self.pooling_function_name = "MaxPool"


class AveragePool(Pool):
    def __init__(self, *args):
        super().__init__(tf.nn.avg_pool, *args)
        self.pooling_function_name = "AveragePool"
