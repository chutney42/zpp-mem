from functools import reduce

import tensorflow as tf

from layer.weight_layer.weight_layer import WeightLayer
from propagator.backward_propagator import Backpropagator, BackwardPropagator
from propagator.direct_propagator import DirectPropagator


class Conv(WeightLayer):
    def __init__(self, filter_dim, stride=[1, 1], number_of_filters=1, padding="SAME", trainable=True,
                 learning_rate=None, momentum=0.0, scope="convoluted_layer"):
        super().__init__(learning_rate, momentum, scope)
        self.stride = [1] + stride + [1]
        self.filter_dim = filter_dim
        self.number_of_filters = number_of_filters
        self.trainable = trainable
        self.padding = padding
        self.output_shape = None
        self.input_flat_shape = None

    def forward_function(self):
        return tf.nn.conv2d

    def update_function(self):
        return tf.nn.conv2d_backprop_filter

    def propagate_function(self):
        return tf.nn.conv2d_backprop_input

    def __str__(self):
        return f"Convolution({self.filter_dim} {self.number_of_filters} {self.stride})"

    def build_forward(self, input_vec, remember_input=True, gather_stats=False):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.save_shape(input_vec)
            self.input_flat_shape = int(
                reduce(lambda x, y: x * y, input_vec.shape[1:]))  # Number of features in input_vec

            if remember_input:
                self.input_vec = input_vec
            (width, length, depth) = input_vec.shape[1], input_vec.shape[2], input_vec.shape[3]
            filter_shape = [self.filter_dim[0], self.filter_dim[1], depth,
                            self.number_of_filters]
            print("Conv")
            print(f"filter_shape:   {filter_shape}")
            filters = tf.get_variable("filters", filter_shape,
                                      initializer=tf.random_normal_initializer())
            print(filters)
            output = self.forward_function()(input_vec, filters, strides=self.stride, padding=self.padding,
                                             name="Convolution")
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
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_vec = self.restore_input()
            filters = tf.get_variable("filters")
            delta_filters = tf.get_variable("delta_weight", filters.shape, initializer=tf.zeros_initializer())

            raw_delta = self.update_function()(input_vec, tf.shape(filters), error, self.stride, self.padding)
            delta_filters = tf.assign(delta_filters, raw_delta + tf.multiply(self.momentum, delta_filters))
            filters = tf.assign(filters, filters - self.learning_rate * delta_filters)
            self.step = filters
            if gather_stats:
                tf.summary.histogram("error", error, family=self.scope)
                tf.summary.histogram("delta", delta_filters, family=self.scope)
                tf.summary.histogram("input", input_vec, family=self.scope)
            return


class DepthWiseConv(Conv):
    def forward_function(self):
        return tf.nn.depthwise_conv2d

    def update_function(self):
        return tf.nn.depthwise_conv2d_native_backprop_filter

    def propagate_function(self):
        return tf.nn.depthwise_conv2d_native_backprop_input


class DepthWiseSeperableConv(Conv):
    def __init__(self, filter_dim, stride=[1, 1], number_of_filters=1, padding="SAME", trainable=True,
                 learning_rate=None, momentum=0.0, scope="depthwise_seperable_conv_layer"):
        super().__init__(filter_dim, stride, number_of_filters, padding, trainable, learning_rate, momentum, scope)

        self.dw_conv = DepthWiseConv(filter_dim, stride, 1, padding, trainable, learning_rate, momentum, scope)
        self.pw_conv = Conv([1, 1], [1, 1], number_of_filters, padding, trainable, learning_rate, momentum, scope)

    def __str__(self):
        return f"DepthWiseSeperableConvolution({self.filter_dim} {self.number_of_filters} {self.stride})"

    def build_forward(self, input_vec, remember_input=True, gather_stats=False):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            print("dwsConv forward")
            print(input_vec)
            input_after_depthwise = self.dw_conv.build_forward(input_vec, remember_input, gather_stats)
            print(f"input after depth   wise: {input_after_depthwise}")
            output = self.pw_conv.build_forward(input_after_depthwise, remember_input, gather_stats)
            print(output)
            return output

    def build_propagate(self, error, gather_stats=False):
        if not self.propagator:
            raise AttributeError("The propagator should be specified")
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if issubclass(self.propagator, DirectPropagator):
                self.pw_conv.propagator = BackwardPropagator()
            else:
                self.pw_conv.propagator = self.propagator

            self.dw_conv.propagator = self.propagator
            pointwise_error = self.pw_conv.build_propagate(error, gather_stats)
            depthwise_error = self.dw_conv.build_propagate(pointwise_error, gather_stats)
            return depthwise_error

    def build_update(self, error, gather_stats=False):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            error_from_pw = self.pw_conv.build_propagate(error)
            self.pw_conv.build_update(error, gather_stats)
            self.dw_conv.build_update(error_from_pw, gather_stats)
            self.step = [self.pw_conv.step, self.dw_conv.step]
            self.pw_conv.step = None
            self.dw_conv.step = None
            return


class ConvManhattan(Conv):
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
