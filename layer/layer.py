import tensorflow as tf


class Layer(object):
    def __init__(self, trainable, scope="layer"):
        self.trainable = trainable
        if trainable:
            self.step = None
        self.scope = scope
        self.input_vec = None
        self.input_shape = None

    def restore_input(self):
        if self.input_vec is None:
            raise AttributeError("Cannot restore input_vec")
        input_vec = self.input_vec
        self.input_vec = None
        return input_vec

    def build_forward(self, input_vec, remember_input=True, gather_stats=True):
        raise NotImplementedError("This method should be implemented in subclass")

    def build_backward(self, error, gather_stats=True):
        raise NotImplementedError("This method should be implemented in subclass")

    def save_shape(self, input_vec):
        self.input_shape = tf.shape(input_vec)

    def restore_shape(self, input_vec):
        return tf.reshape(input_vec, self.input_shape)

    def flatten_input(self, input_vec):
        return tf.layers.Flatten()(input_vec)
