import tensorflow as tf

from layer import WeightLayer
from layer.layer import Layer


class BatchNormalization(Layer):

    def __init__(self, axis=-1, momentum=0.99, scope="batch_normalization_layer"):
        super().__init__(trainable=True, scope=scope)
        self.axis = axis
        self.momentum = momentum

    def build_forward(self, input, remember_input=False, gather_stats=False):
        with tf.name_scope(self.scope):
            if remember_input:
                self.save_input(input)

            output = tf.layers.batch_normalization(input, axis=self.axis, momentum=self.momentum,
                                                   training=self.traning_mode)
            if gather_stats:
                tf.summary.histogram("input", input, family=self.scope)
                tf.summary.histogram("output", output, family=self.scope)

        return output

    def __str__(self):
        return f"BatchNormalization({self.axis}, {self.momentum})"
