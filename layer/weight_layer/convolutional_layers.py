from functools import reduce

import tensorflow as tf

from layer.weigh_layer import WeightLayer


class ConvolutionalLayer(WeightLayer):
    def __init__(self, filter_dim, stride=[1, 1, 1, 1], number_of_filters=1, padding="SAME",
                 trainable=True, learning_rate=0.5,
                 scope="convoluted_layer"):
        super().__init__(learning_rate, scope)
        self.stride = stride
        self.filter_dim = filter_dim
        self.number_of_filters = number_of_filters
        self.trainable = trainable
        self.padding = padding
        self.output_shape = None
        self.input_flat_shape = None

    def build_forward(self, input_vec, remember_input=True, gather_stats=True):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.save_shape(input_vec)
            self.input_flat_shape= int(reduce(lambda x, y: x * y, input_vec.shape[1:]))  # Number of features in input_vec

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
            backprop_error = self.propagator.propagate_conv(self,error)
            return self.restore_shape(backprop_error)

    def build_update(self, error, gather_stats=True):
        input_vec = self.restore_input()
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            filters = tf.get_variable("filters")
            delta_filters = tf.nn.conv2d_backprop_filter(input_vec, tf.shape(filters), error,
                                                                self.stride, self.padding)
            filters = tf.assign(filters, filters - self.learning_rate * delta_filters)
            self.step = filters
            return

