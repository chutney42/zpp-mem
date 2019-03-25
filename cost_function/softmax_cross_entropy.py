import tensorflow as tf
from cost_function.cost_function import CostFunction
from layer.util_layer.softmax import Softmax


class SoftmaxCrossEntropy(CostFunction):
    @staticmethod
    def cost(logits, labels, name=None):
        with tf.name_scope(name or "softmax_cross_entropy_value"):
            return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    @staticmethod
    def error(predictions, labels, name=None):
        with tf.name_scope(name or "softmax_cross_entropy_delta"):
            return tf.subtract(predictions, labels)
