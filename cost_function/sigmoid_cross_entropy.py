import tensorflow as tf
from cost_function.cost_function import CostFunction
from layer.activation.activation_layer import Sigmoid


class SigmoidCrossEntropy(CostFunction):
    @staticmethod
    def cost(logits, labels, name=None):
        with tf.name_scope(name or "sigmoid_cross_entropy_value"):
            return tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=output)
                
    @staticmethod
    def error(predictions, labels, name=None):
        with tf.name_scope(name or "sigmoid_cross_entropy_delta"):
            return tf.subtract(predictions, labels)
