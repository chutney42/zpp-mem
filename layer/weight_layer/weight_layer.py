from layer.layer import Layer
import tensorflow as tf


class WeightLayer(Layer):
    def __init__(self, func=None, scope="weight_layer"):
        super().__init__(trainable=True, scope=scope)
        self.func = func

    def build_propagate(self, error, output):
        input = self.restore_input()
        return tf.gradients(output, input, error)[0]

    def build_update(self, error, output, optimizer):
        if not self.variables:
            return
        gradients = tf.gradients(output, self.variables, error)
        grads_and_vars = zip(gradients, self.variables)
        self.step = optimizer.apply_gradients(grads_and_vars)
        return
