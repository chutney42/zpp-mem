import tensorflow as tf

from propagator.propagator import Propagator


class DirectPropagator(Propagator):
    def __init__(self, output_error_dim):
        self.output_error_dim = output_error_dim


class DirectFixedRandom(DirectPropagator):
    def __init__(self, output_error_dim, initializer=tf.random_normal_initializer()):
        self.initializer = initializer
        super().__init__(output_error_dim)

    def propagate_fc(self, layer, error):
        weights = tf.get_variable("weights")
        propagator = self.get_weights(weights)
        return layer.restore_shape(tf.matmul(error, propagator))

    def propagate_conv(self, layer, error):
        filters = self.get_filter(layer.input_flat_shape)
        return layer.restore_shape(tf.matmul(error, filters))

    def get_weights(self, weights):
        return tf.get_variable("direct_random_weights", shape=[self.output_error_dim, weights.shape[0]],
                               initializer=self.initializer)

    def get_filter(self, dim):
        return tf.get_variable("direct_random_weights", shape=[self.output_error_dim, dim],
                               initializer=self.initializer)