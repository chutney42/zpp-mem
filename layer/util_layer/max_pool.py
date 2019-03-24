import tensorflow as tf

from layer.layer import Layer


class MaxPool(Layer):
    def __init__(self, ksize, strides, padding, data_format='NHWC', scope="max_pool_layer"):
        super().__init__(trainable=False, scope=scope)
        # self.dim = dim
        # self.input_vec = input_vec
        self.ksize = [-1, -1] + +ksize
        self.strides = [0, 0] + +strides
        self.padding = padding
        self.data_format = data_format

    def build_forward(self, input_vec, remember_input=True, gather_stats=True):
        if remember_input:
            self.input_vec = input_vec

        with tf.variable_scope(self.scope, tf.AUTO_REUSE):
            (width, length, depth) = input_vec.shape[1], input_vec.shape[2], input_vec.shape[3]
            output = tf.nn.max_pool(input_vec, self.ksize, self.strides, self.padding, self.data_format)
            return output

    def build_backward(self, error, gather_stats=True):
        input_vec = self.restore_input()
        with tf.variable_scope(self.scope):
            xs = input_vec
            ys = tf.nn.max_pool(input_vec, self.ksize, self.strides, self.padding, self.data_format)
            backprop_error = tf.gradients(ys, xs, error)
            return backprop_error
