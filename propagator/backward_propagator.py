import tensorflow as tf

from propagator.propagator import Propagator


class BackwardPropagator(Propagator):
    def propagate_fc(self, layer, error):
        weights = tf.get_variable("weights")
        propagator = self.get_weights(weights)
        return tf.matmul(error, propagator)

    def propagate_conv(self, layer, error):
        filters = tf.get_variable("filters")
        filters = self.get_filter(filters)
        backprop_error = layer.propagate_function()(layer.input_shape, filters, error, layer.stride,
                                                     layer.padding)
        return backprop_error


class Backpropagator(BackwardPropagator):
    def get_filter(self, filters):
        return filters

    def get_weights(self, weights):
        return tf.transpose(weights)


class FixedRandom(BackwardPropagator):
    def __init__(self, initializer=tf.random_normal_initializer()):
        super().__init__()
        self.initializer = initializer

    def get_filter(self, filters):
        return tf.get_variable("random_filters", shape=filters.get_shape().as_list(),
                               initializer=self.initializer)

    def get_weights(self, weights):
        return tf.get_variable("random_weights", shape=tf.transpose(weights).get_shape().as_list(),
                               initializer=self.initializer)
