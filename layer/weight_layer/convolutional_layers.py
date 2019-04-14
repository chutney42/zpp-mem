from functools import reduce

import tensorflow as tf
from tensorflow.initializers import he_normal
from layer.weight_layer.weight_layer import WeightLayer


class ConvolutionalLayer(WeightLayer):
    def __init__(self, filter_dim, stride=[1, 1], number_of_filters=1, padding="SAME", trainable=True,
                 learning_rate=None, momentum=0.0, scope="convoluted_layer", filters_initializer=he_normal):
        super().__init__(learning_rate, momentum, scope)
        self.filters_initializer = filters_initializer
        self.stride = [1] + stride + [1]
        self.filter_dim = filter_dim
        self.number_of_filters = number_of_filters
        self.trainable = trainable
        self.padding = padding
        self.output_shape = None
        self.input_flat_shape = None

    def __str__(self):
        return f"ConvolutionalLayer({self.filter_dim} {self.number_of_filters} {self.stride})"

    def build_forward(self, input_vec, remember_input=True, gather_stats=False):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.save_shape(input_vec)
            self.input_flat_shape = int(
                reduce(lambda x, y: x * y, input_vec.shape[1:]))  # Number of features in input_vec

            if remember_input:
                self.input_vec = input_vec
            (width, length, depth) = input_vec.shape[1], input_vec.shape[2], input_vec.shape[3]
            filter_shape = [self.filter_dim[0], self.filter_dim[1], depth, self.number_of_filters]
            filters = tf.get_variable("filters", filter_shape, initializer=self.filters_initializer)
            output = tf.nn.conv2d(input_vec, filters, strides=self.stride, padding=self.padding, name="Convolution")
            self.output_shape = tf.shape(output)

            if gather_stats:
                tf.summary.histogram("weights", filters, family=self.scope)
                tf.summary.histogram("output", output, family=self.scope)

            return output

    def build_propagate(self, error, gather_stats=False):
        if not self.propagator:
            raise AttributeError("The propagator should be specified")
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            backprop_error = self.propagator.propagate_conv(self, error)
            return self.restore_shape(backprop_error)

    def build_update(self, error, gather_stats=False):
        input_vec = self.restore_input()
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            filters = tf.get_variable("filters")
            delta_filters = tf.get_variable("delta_weight", filters.shape, initializer=tf.zeros_initializer())

            raw_delta = tf.nn.conv2d_backprop_filter(input_vec, tf.shape(filters), error, self.stride, self.padding)
            delta_filters = tf.assign(delta_filters, raw_delta + tf.multiply(self.momentum, delta_filters))
            filters = tf.assign(filters, filters - self.learning_rate * delta_filters)
            self.step = filters
            if gather_stats:
                tf.summary.histogram("error", error, family=self.scope)
                tf.summary.histogram("delta", delta_filters, family=self.scope)
                tf.summary.histogram("input", input_vec, family=self.scope)
            return


class ConvolutionalLayerManhattan(ConvolutionalLayer):
    def build_update(self, error, gather_stats=False):
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
                tf.summary.histogram("error", error, family=self.scope)
                tf.summary.histogram("delta", delta_filters, family=self.scope)
                tf.summary.histogram("manhattan", manhattan, family=self.scope)
                tf.summary.histogram("weights", filters, family=self.scope)
                tf.summary.histogram("input", input_vec, family=self.scope)
            return
