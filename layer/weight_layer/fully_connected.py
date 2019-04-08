import tensorflow as tf

from layer.weight_layer.weight_layer import WeightLayer


class FullyConnected(WeightLayer):
    def __init__(self, output_dim, learning_rate=None, momentum=0.0, scope="fully_connected_layer", flatten=False):
        super().__init__(learning_rate, momentum, scope)
        self.output_dim = output_dim
        self.flatten = flatten

    def __str__(self):
        return f"FullyConnected({self.output_dim})"

    def build_forward(self, input_vec, remember_input=True, gather_stats=False):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.save_shape(input_vec)
            if self.flatten:
                input_vec = tf.layers.Flatten()(input_vec)
            if remember_input:
                self.input_vec = input_vec
            weights = tf.get_variable("weights", [input_vec.shape[1], self.output_dim],
                                      initializer=tf.random_normal_initializer())
            biases = tf.get_variable("biases", [self.output_dim],
                                     initializer=tf.constant_initializer())

            output = tf.add(tf.matmul(input_vec, weights), biases)
            if gather_stats:
                tf.summary.histogram("input", input_vec, family=self.scope)
                tf.summary.histogram("weights", weights, family=self.scope)
                tf.summary.histogram("biases", biases, family=self.scope)
                tf.summary.histogram("output", output, family=self.scope)
            return output

    def build_propagate(self, error, gather_stats=False):
        if not self.propagator:
            raise AttributeError("The propagator should be specified")
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            backprop_error = self.propagator.propagate_fc(self, error)
            if self.flatten:
                backprop_error = tf.reshape(backprop_error, self.input_shape)
            return backprop_error

    def build_update(self, error, gather_stats=False):
        input_vec = self.restore_input()
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            delta_weights = tf.get_variable("delta_weights", weights.shape, initializer=tf.zeros_initializer())
            delta_biases = tf.get_variable("delta_biases", biases.shape, initializer=tf.zeros_initializer())

            raw_delta = tf.matmul(tf.transpose(input_vec), error)
            delta_weights = tf.assign(delta_weights, raw_delta + tf.multiply(self.momentum, delta_weights))
            delta_biases = tf.assign(delta_biases,
                                     tf.reduce_mean(error, axis=[0]) + tf.multiply(self.momentum, delta_biases))

            weights = tf.assign(weights, tf.subtract(weights, tf.multiply(self.learning_rate, delta_weights)))
            biases = tf.assign(biases, tf.subtract(biases, tf.multiply(self.learning_rate, delta_biases)))
            self.step = (weights, biases)

            if gather_stats:
                tf.summary.histogram("error", error, family=self.scope)
                tf.summary.histogram("delta_weights", delta_weights, family=self.scope)
                tf.summary.histogram("delta_biases", delta_biases, family=self.scope)
            return


class FullyConnectedManhattan(FullyConnected):

    def build_update(self, error, gather_stats=False):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_vec = self.restore_input()
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            delta_weights = tf.get_variable("delta_weights", weights.shape, initializer=tf.zeros_initializer())
            delta_biases = tf.get_variable("delta_biases", biases.shape, initializer=tf.zeros_initializer())

            raw_delta = tf.matmul(tf.transpose(input_vec), error)
            manhattan = tf.sign(raw_delta)
            delta_weights = tf.assign(delta_weights, manhattan + tf.multiply(self.momentum, delta_weights))
            delta_biases = tf.assign(delta_biases,
                                     tf.reduce_mean(error, axis=[0]) + tf.multiply(self.momentum, delta_biases))

            weights = tf.assign(weights, tf.subtract(weights, tf.multiply(self.learning_rate, delta_weights)))
            biases = tf.assign(biases, tf.subtract(biases, tf.multiply(self.learning_rate, delta_biases)))
            self.step = (weights, biases)
            if gather_stats:
                if gather_stats:
                    tf.summary.histogram("error", error, family=self.scope),
                    tf.summary.histogram("manhattan", manhattan, family=self.scope)
                    tf.summary.histogram("delta_weights", delta_weights, family=self.scope)
                    tf.summary.histogram("delta_biases", delta_biases, family=self.scope)
            return
