import tensorflow as tf

from layer import WeightLayer
from layer.layer import Layer


class BatchNormalization(Layer):
    def __init__(self, trainable=True, axis=-1, momentum=0.99, scope="batch_normalization_layer"):
        super().__init__(trainable, scope=scope)
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
                tf.summary.histogram("output", output, family=self.scope
                                     )
            return output

    def gather_stats_backward(self, gradients):
        tf.summary.histogram("delta_input", gradients[0], family=self.scope)
        tf.summary.histogram("delta_1", gradients[1], family=self.scope)
        tf.summary.histogram("delta_2", gradients[2], family=self.scope)
