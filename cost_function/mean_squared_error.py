import tensorflow as tf
from cost_function.cost_function import CostFunction


class MeanSquaredError(CostFunction):
    @staticmethod
    def cost(predictions, labels, name=None):
        with tf.name_scope(name or "mean_squared_error_value"):
            return tf.losses.mean_squared_error(labels, predictions)

    @staticmethod
    def error(predictions, labels, name=None):
        with tf.name_scope(name or "mean_squared_error_delta"):
            return tf.subtract(predictions, labels)
