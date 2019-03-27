import tensorflow as tf
from layer.layer import Layer


class Softmax(Layer):
    def __init__(self, softmax_cross_entropy=False, scope="softmax_layer"):
        super().__init__(trainable=False, scope=scope)
        self.softmax_cross_entropy = softmax_cross_entropy

    def __str__(self):
        return "Softmax()"

    def build_forward(self, input_vec, remember_input=True, gather_stats=True):
        if remember_input:
            self.input_vec = input_vec
        return tf.nn.softmax(input_vec, name=self.scope)

    def build_backward(self, error, gather_stats=True):
        input_vec = self.restore_input()
        with tf.name_scope(self.scope):
            if self.softmax_cross_entropy:
                return error
            return tf.gradients(tf.nn.softmax(input_vec), input_vec, error)[0]
