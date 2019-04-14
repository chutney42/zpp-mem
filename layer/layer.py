import tensorflow as tf


class Layer(object):
    def __init__(self, trainable, scope="layer"):
        self.trainable = trainable
        if trainable:
            self.step = None
            self.variables = []
        self.scope = scope
        self.input = None
        self.input_shape = None
        self.learning_rate = None
        self.momentum = None

    def save_input(self, input):
        self.input = input

    def restore_input(self):
        if self.input is None:
            raise AttributeError("No input to restore")
        input = self.input
        self.input = None
        return input

    def gather_stats_backward(self, gradients):
        raise NotImplementedError("This method should be implemented in subclass")
    
    def build_forward(self, input, remember_input=False, gather_stats=True):
        raise NotImplementedError("This method should be implemented in subclass")

    def build_backward(self, error, output, optimizer, gather_stats=True):
        variables = [self.restore_input()] + self.variables
        gradients = tf.gradients(output, variables, error)
        if self.variables: 
            grads_and_vars = zip(gradients[1:], variables[1:])
            self.step = optimizer.apply_gradients(grads_and_vars)
        if gather_stats:
            self.gather_stats_backward(gradients)
        return gradients[0]
