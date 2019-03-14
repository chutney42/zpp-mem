import tensorflow as tf


class Propagator(object):
    def propagate_conv(self, layer, error):
        raise NotImplementedError("This method should be implemented in subclass")

    def propagate_fc(self, layer, error):
        raise NotImplementedError("This method should be implemented in subclass")

    def __get_filter(self, filters):
        raise NotImplementedError("This method should be implemented in subclass")

    def __get_weights(self, weights):
        raise NotImplementedError("This method should be implemented in subclass")


class BackwardPropagator(Propagator):
    def propagate_fc(self, layer, error):
        weights = tf.get_variable("weights")
        propagator = self.__get_weights(weights)
        return layer.restore_shape(tf.matmul(error, propagator))

    def propagate_conv(self, layer, error):
        filters = tf.get_variable("filters")
        filters = self.__get_filter(filters)
        backprop_error = tf.nn.conv2d_backprop_input(layer.input_shape, filters, error, layer.stride,
                                                     layer.padding)
        return backprop_error


class Backpropagator(BackwardPropagator):
    def __get_filter(self, filters):
        return filters

    def __get_weights(self, weights):
        return tf.transpose(weights)


class FixedRandom(BackwardPropagator):
    def __init__(self, initializer=tf.random_normal_initializer()):
        super().__init__()
        self.initializer = initializer

    def __get_filter(self, filters):
        return tf.get_variable("random_filters", shape=filters.get_shape().as_list(),
                               initializer=self.initializer)

    def __get_weights(self, weights):
        return tf.get_variable("random_weights", shape=tf.transpose(weights).get_shape().as_list(),
                               initializer=self.initializer)


class DirectPropagator(Propagator):
    def __init__(self, output_error_dim):
        self.output_error_dim = output_error_dim


class DirectFixedRandom(DirectPropagator):
    def __init__(self, output_error_dim, initializer=tf.random_normal_initializer()):
        self.initializer = initializer
        super().__init__(output_error_dim)

    def propagate_fc(self, layer, error):
        weights = tf.get_variable("weights")
        propagator = self.__get_weights(weights)
        return layer.restore_shape(tf.matmul(error, propagator))

    def propagate_conv(self, layer, error):
        filters = self.__get_filter(layer.input_flat_shape)
        return layer.restore_shape(tf.matmul(error, filters))

    def __get_weights(self, weights):
        return tf.get_variable("direct_random_weights", shape=[self.output_error_dim, weights.shape[0]],
                               initializer=self.initializer)

    def __get_filter(self, dim):
        return tf.get_variable("direct_random_weights", shape=[self.output_error_dim, dim],
                               initializer=self.initializer)
