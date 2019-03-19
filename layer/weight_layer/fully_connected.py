import tensorflow as tf

from layer.weight_layer.weight_layer import WeightLayer


class FullyConnected(WeightLayer):
    def __init__(self, output_dim, learning_rate=0.5, momentum=0.0, scope="fully_connected_layer"):
        super().__init__(learning_rate, momentum, scope)
        self.output_dim = output_dim

    def __str__(self):
        return f"FullyConnected({self.output_dim})"

    def build_forward(self, input_vec, remember_input=True, gather_stats=True):
        self.save_shape(input_vec)
        input_vec = self.flatten_input(input_vec)
        if remember_input:
            self.input_vec = input_vec
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights", [input_vec.shape[1], self.output_dim],
                                      initializer=tf.random_normal_initializer())
            biases = tf.get_variable("biases", [self.output_dim],
                                     initializer=tf.constant_initializer())
            return tf.add(tf.matmul(input_vec, weights), biases)

    def build_propagate(self, error, gather_stats=True):
        if not self.propagator:
            raise AttributeError("The propagator should be specified")
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            return self.propagator.propagate_fc(self, error)

    def build_update(self, error, gather_stats=True):
        input_vec = self.restore_input()
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            delta_weights = tf.get_variable("delta_weights", weights.shape, initializer=tf.zeros_initializer())
            delta_biases = tf.get_variable("delta_biases", biases.shape, initializer=tf.zeros_initializer())

            raw_delta = tf.matmul(tf.transpose(input_vec), error)
            delta_weights = tf.assign(delta_weights, raw_delta + tf.multiply(self.momentum, delta_weights))
            delta_biases = tf.assign(delta_biases, tf.reduce_mean(error, axis=[0]) + tf.multiply(self.momentum, delta_biases))

            weights = tf.assign(weights, tf.subtract(weights, tf.multiply(self.learning_rate, delta_weights)))
            biases = tf.assign(biases, tf.subtract(biases, tf.multiply(self.learning_rate, delta_biases)))
            self.step = (weights, biases)

            if gather_stats:
                tf.summary.image(f"weights_{self.scope}", tf.reshape(weights, (1, weights.shape[0], weights.shape[1], 1)))
            return


class FullyConnectedManthattan(FullyConnected):

    def build_update(self, error, gather_stats=True):
        input_vec = self.restore_input()
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            delta_weights = tf.get_variable("delta_weights", weights.shape, initializer=tf.zeros_initializer())
            delta_biases = tf.get_variable("delta_biases", biases.shape, initializer=tf.zeros_initializer())

            raw_delta = tf.matmul(tf.transpose(input_vec), error)
            manhattan = tf.sign(raw_delta)
            delta_weights = tf.assign(delta_weights, manhattan + tf.multiply(self.momentum, delta_weights))
            delta_biases = tf.assign(delta_biases, error + tf.multiply(self.momentum, delta_biases))

            weights = tf.assign(weights, tf.subtract(weights, tf.multiply(self.learning_rate, delta_weights)))
            biases = tf.assign(biases, tf.subtract(biases, tf.multiply(self.learning_rate, tf.reduce_mean(delta_biases,
                                                                                                          axis=[0]))))
            self.step = (weights, biases)
            if gather_stats:
                tf.summary.image(f"weights_{self.scope}", tf.reshape(weights, (1, weights.shape[0], weights.shape[1], 1)))
            return
