import tensorflow as tf
from cost_function.cost_function import CostFunction
from layer.util_layer.softmax import Softmax


class SoftmaxCrossEntropy(CostFunction):
    @staticmethod
    def cost(logits, labels, scope="softmax_cross_entropy_value"):
        with tf.name_scope(scope):
            return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    @staticmethod
    def error(predictions, labels, scope="softmax_cross_entropy_delta"):
        with tf.name_scope(scope):
            return tf.subtract(predictions, labels)
