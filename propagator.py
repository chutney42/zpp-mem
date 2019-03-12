import tensorflow as tf


class Propagator(object):
    def get_fc(self, weights):
        raise NotImplementedError("This method should be implemented in subclass")

    def get_conv(self, weights):
        raise NotImplementedError("This method should be implemented in subclass")


class Backpropagator(Propagator):
    def get_fc(self, weights):
        return tf.transpose(weights)

    def get_conv(self, filters):
        return filters


class FixedRandom(Propagator):
    def __init__(self, initializer=tf.random_normal_initializer()):
        self.initializer = initializer

    def get_fc(self, weights):
        return tf.get_variable("random_weights", shape=tf.transpose(weights).get_shape().as_list(),
            initializer=self.initializer)

    def get_conv(self, filters):
        return tf.get_variable("random_filters", shape=filters.get_shape().as_list(),
            initializer=self.initializer)


class DirectPropagator(Propagator):
    def __init__(self, output_error_dim):
        self.output_error_dim = output_error_dim


class DirectFixedRandom(DirectPropagator):
    def __init__(self, output_error_dim, initializer=tf.random_normal_initializer()):
        self.initializer = initializer
        super().__init__(output_error_dim)

    def get_fc(self, weights):
        return tf.get_variable("random_weights", shape=[self.output_error_dim, weights.shape[0]],
            initializer=self.initializer)

    def get_conv(self, dim):
        return tf.get_variable("random_weights", shape=[self.output_error_dim, dim],
                               initializer=self.initializer)
