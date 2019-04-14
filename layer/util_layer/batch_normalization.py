import tensorflow as tf
from layer.layer import Layer


class BatchNormalization(Layer):
    def __init__(self, training=True, axis=-1, momentum=0.99, scope="batch_normalization_layer"):
        super().__init__(trainable=True, scope=scope)
        self.training = training
        self.axis = axis
        self.momentum = momentum

    def build_forward(self, input, remember_input=False, gather_stats=False):
        if remember_input:
            save_input(input)

        with tf.name_scope(self.scope):
            output = tf.layers.batch_normalization(input, axis=self.axis, momentum=self.momentum,
                                                   training=self.training)

        if gather_stats:
            tf.summary.histogram("input", input, family=self.scope)
            tf.summary.histogram("output", output, family=self.scope)
