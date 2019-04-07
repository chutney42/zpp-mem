import tensorflow as tf
from layer.layer import Layer


def sigmoid_prime(x, name=None):
    with tf.name_scope(name or "sigmoid_prime"):
        return tf.multiply(tf.sigmoid(x), (tf.constant(1.0) - tf.sigmoid(x)))


def tanh_prime(x, name=None):
    with tf.name_scope(name or "tanh_prime"):
        return tf.constant(1.0) - tf.multiply(tf.tanh(x), tf.tanh(x))


def relu_prime(x, name=None):
    with tf.name_scope(name or "relu_prime"):
        condition = tf.less_equal(x, 0)
        return tf.where(condition, tf.zeros_like(x), tf.ones_like(x))


def leaky_relu_prime(x, alpha=0.3, name=None):
    with tf.name_scope(name or "leaky_relu_prime"):
        condition = tf.less_equal(x, 0)
        return tf.where(condition, tf.multiply(tf.ones_like(x), alpha), tf.ones_like(x))


class ActivationLayer(Layer):
    def __init__(self, func, func_prime, scope="activation_function_layer"):
        super().__init__(trainable=False, scope=scope)
        self.func = func
        self.func_prime = func_prime

    def build_forward(self, input_vec, remember_input=False, gather_stats=False):
        if remember_input:
            self.input_vec = input_vec
        with tf.variable_scope(self.scope, tf.AUTO_REUSE):
            return self.func(input_vec)

    def build_backward(self, error, gather_stats=False):
        input_vec = self.restore_input()
        with tf.variable_scope(self.scope):
            activated =  self.func_prime(input_vec)
            result = tf.multiply(error, activated)
            if gather_stats:
                tf.summary.histogram("input", input_vec)
                tf.summary.histogram("activated", activated)
                tf.summary.histogram("propagated_error", result)
            return result


class Sigmoid(ActivationLayer):
    def __init__(self, sigmoid_cross_entropy=False, scope="sigmoid_layer"):
        super().__init__(tf.sigmoid, sigmoid_prime, scope=scope)
        self.sigmoid_cross_entropy = sigmoid_cross_entropy

    def __str__(self):
        return "Sigmoid()"

    def build_backward(self, error, gather_stats=False):
        input_vec = self.restore_input()
        with tf.variable_scope(self.scope):
            if gather_stats:
                tf.summary.histogram("input", input_vec)
            if self.sigmoid_cross_entropy:
                if gather_stats:
                     tf.summary.histogram("propagated_error", error)
                return error
            activated = self.func_prime(input_vec)
            result = tf.multiply(error, activated)
            if gather_stats:
                tf.summary.histogram("activated", activated)
                tf.summary.histogram("propagated_error", result)
            return result


class Tanh(ActivationLayer):
    def __init__(self, scope="tanh_layer"):
        super().__init__(tf.tanh, tanh_prime, scope=scope)

    def __str__(self):
        return "Tanh()"


class ReLu(ActivationLayer):
    def __init__(self, scope="relu_layer"):
        super().__init__(tf.nn.relu, relu_prime, scope=scope)

    def __str__(self):
        return "ReLu()"


class LeakyReLu(ActivationLayer):
    def __init__(self, scope="leaky_relu_layer"):
        super().__init__(tf.nn.leaky_relu, leaky_relu_prime, scope=scope)

    def __str__(self):
        return "LeakyReLu()"
