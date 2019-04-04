from functools import reduce

import tensorflow as tf
from layer.block import Block
from layer.activation import ReLu
from layer.util_layer import BatchNormalization, MaxPool
from layer.weight_layer.weight_layer import WeightLayer


class ConvolutionalLayer(WeightLayer):
    def __init__(self, filter_dim, stride=[1, 1], number_of_filters=1, padding="SAME", trainable=True,
                 learning_rate=0.5, momentum=0.0, scope="convoluted_layer"):
        super().__init__(learning_rate, momentum, scope)
        self.stride = [1] + stride + [1]
        self.filter_dim = filter_dim
        self.number_of_filters = number_of_filters
        self.trainable = trainable
        self.padding = padding
        self.output_shape = None
        self.input_flat_shape = None

    def __str__(self):
        return f"ConvolutionalLayer({self.filter_dim} {self.number_of_filters} {self.stride})"

    def build_forward(self, input_vec, remember_input=True, gather_stats=True):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.save_shape(input_vec)
            self.input_flat_shape = int(
                reduce(lambda x, y: x * y, input_vec.shape[1:]))  # Number of features in input_vec

            if remember_input:
                self.input_vec = input_vec
            (width, length, depth) = input_vec.shape[1], input_vec.shape[2], input_vec.shape[3]
            filter_shape = [self.filter_dim[0], self.filter_dim[1], depth,
                            self.number_of_filters]
            filters = tf.get_variable("filters", filter_shape,
                                      initializer=tf.random_normal_initializer())
            output = tf.nn.conv2d(input_vec, filters, strides=self.stride, padding=self.padding, name="Convolution")
            self.output_shape = tf.shape(output)
            return output

    def build_propagate(self, error, gather_stats=True):
        if not self.propagator:
            raise AttributeError("The propagator should be specified")
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            backprop_error = self.propagator.propagate_conv(self, error)
            return self.restore_shape(backprop_error)

    def build_update(self, error, gather_stats=True):
        input_vec = self.restore_input()
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            filters = tf.get_variable("filters")
            delta_filters = tf.get_variable("delta_weight", filters.shape, initializer=tf.zeros_initializer())

            raw_delta = tf.nn.conv2d_backprop_filter(input_vec, tf.shape(filters), error, self.stride, self.padding)
            delta_filters = tf.assign(delta_filters, raw_delta + tf.multiply(self.momentum, delta_filters))
            filters = tf.assign(filters, filters - self.learning_rate * delta_filters)
            self.step = filters
            if gather_stats:
                tf.summary.image(f"delta", put_kernels_on_grid(raw_delta), 1)
                tf.summary.image(f"delta", put_kernels_on_grid(delta_filters), 1)

                tf.summary.image(f"filters", put_kernels_on_grid(filters), 1)
            return


class ConvolutionalLayerManhattan(ConvolutionalLayer):
    def build_update(self, error, gather_stats=True):
        input_vec = self.restore_input()
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            filters = tf.get_variable("filters")
            delta_filters = tf.get_variable("delta_weight", filters.shape, initializer=tf.zeros_initializer())

            raw_delta = tf.nn.conv2d_backprop_filter(input_vec, tf.shape(filters), error, self.stride, self.padding)
            manhattan = tf.sign(raw_delta)
            delta_filters = tf.assign(delta_filters, manhattan + tf.multiply(self.momentum, delta_filters))
            filters = tf.assign(filters, filters - self.learning_rate * delta_filters)
            self.step = filters
            if gather_stats:
                tf.summary.image(f"delta", put_kernels_on_grid(raw_delta), 1)
                tf.summary.image(f"manhattan", put_kernels_on_grid(manhattan), 1)
                tf.summary.image(f"filters", put_kernels_on_grid(filters), 1)
            return


def put_kernels_on_grid(kernel, grid_Y=None, grid_X=None, pad=1):
    def factorization(n):
        from numpy.ma import sqrt
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                return i, int(n / i)

    if grid_Y is None:
        (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel = (kernel - x_min) / (x_max - x_min)

    x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad
    channels = kernel.get_shape()[2]
    x = tf.transpose(x, (3, 0, 1, 2))
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))
    x = tf.transpose(x, (0, 2, 1, 3))
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))
    x = tf.transpose(x, (2, 1, 3, 0))
    x = tf.transpose(x, (2, 0, 1, 3))
    return tf.image.convert_image_dtype(x, dtype=tf.uint8)
