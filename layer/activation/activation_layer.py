import tensorflow as tf
from layer.layer import Layer


class ActivationLayer(Layer):
    def __init__(self, func, scope="activation_function_layer"):
        super().__init__(trainable=False, scope=scope)
        self.func = func

    def build_forward(self, input, remember_input=False, gather_stats=True):
        if remember_input:
            self.save_input(input)
        with tf.name_scope(self.scope):
            output = self.func(input)
        if gather_stats:
            tf.summary.histogram("pre_activation", input)
            tf.summary.histogram("post_activation", output)
        return output

    def gather_stats_backward(self, gradients):
        pass


class Sigmoid(ActivationLayer):
    def __init__(self, scope="sigmoid_layer"):
        super().__init__(tf.sigmoid, scope=scope)
    
    def __str__(self):
        return "Sigmoid()"

    
class Tanh(ActivationLayer):
    def __init__(self, scope="tanh_layer"):
        super().__init__(tf.tanh, scope=scope)

    def __str__(self):
        return "Tanh()"


class ReLu(ActivationLayer):
    def __init__(self, scope="relu_layer"):
        super().__init__(tf.nn.relu, scope=scope)

    def __str__(self):
        return "ReLu()"


class LeakyReLu(ActivationLayer):
    def __init__(self, alpha=0.2, scope="leaky_relu_layer"):
        super().__init__(lambda x: tf.nn.leaky_relu(x, alpha), scope=scope)

    def __str__(self):
        return "LeakyReLu()"


class Softmax(ActivationLayer):
    def __init__(self, scope="softmax_layer"):        
        super().__init__(tf.nn.softmax, scope=scope)

    def __str__(self):
        return "Softmax()"
