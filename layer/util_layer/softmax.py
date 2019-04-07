import tensorflow as tf
from layer.layer import Layer


class Softmax(Layer):
    def __init__(self, softmax_cross_entropy=False, scope="softmax_layer"):
        super().__init__(trainable=False, scope=scope)
        self.softmax_cross_entropy = softmax_cross_entropy

    def __str__(self):
        return "Softmax()"

    def build_forward(self, input_vec, remember_input=True, gather_stats=False):
        if remember_input:
            self.input_vec = input_vec
        output = tf.nn.softmax(input_vec, name=self.scope)
        if gather_stats:
            tf.summary.histogram("output", output, family=self.scope)
        return output

    def build_backward(self, error, gather_stats=False):
        with tf.name_scope(self.scope):
            input_vec = self.restore_input()
            if gather_stats:
                tf.summary.histogram("error", error, family=self.scope)
            if self.softmax_cross_entropy:
                return error
            delta = tf.gradients(tf.nn.softmax(input_vec), input_vec, error)[0]
            if gather_stats:
                tf.summary.histogram("delta", delta, family=self.scope)
            return delta
