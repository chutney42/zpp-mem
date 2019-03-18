import tensorflow as tf

from layer.layer import Layer


class Flatten(Layer):
    def __init__(self, scope="flatten_layer"):
        super().__init__(trainable=False, scope=scope)

    def __str__(self):
        return "Flatten()"
    def build_forward(self, input_vec, remember_input=False, gather_stats=True):
        self.save_shape(input_vec)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_vec = tf.layers.Flatten()(input_vec)
            return input_vec

    def build_backward(self, error, gather_stats=True):
        with tf.variable_scope(self.scope, reuse=True):
            return tf.reshape(error, self.input_shape)
