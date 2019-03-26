import tensorflow as tf
from cost_function.cost_function import CostFunction
from layer.activation.activation_layer import Sigmoid


class SigmoidCrossEntropy(CostFunction):
    @staticmethod
    def cost(logits, labels, scope="sigmoid_cross_entropy_value"):
        with tf.name_scope(scope):
            return tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=output)
                
    @staticmethod
    def error(predictions, labels, scope="sigmoid_cross_entropy_delta"):
        with tf.name_scope(scope):
            return tf.subtract(predictions, labels)
