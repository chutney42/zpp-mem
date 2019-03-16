import tensorflow as tf

from layer.weight_layer import WeightLayer


class FullyConnected(WeightLayer):
    def __init__(self, output_dim, learning_rate=0.5, scope="fully_connected_layer"):
        super().__init__(learning_rate, scope)
        self.output_dim = output_dim

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
            return self.propagator.propagate_fc(self,error)

    def build_update(self, error, gather_stats=True):
        input_vec = self.restore_input()
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")
            delta_biases = error
            delta_weights = tf.matmul(tf.transpose(input_vec), error)
            weights = tf.assign(weights, tf.subtract(weights, tf.multiply(self.learning_rate, delta_weights)))
            biases = tf.assign(biases, tf.subtract(biases, tf.multiply(self.learning_rate, tf.reduce_mean(delta_biases,
                axis=[0]))))
            self.step = (weights, biases)
            return