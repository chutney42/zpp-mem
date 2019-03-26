import tensorflow as tf
from cost_function.cost_function import CostFunction


class MeanSquaredError(CostFunction):
    @staticmethod
    def cost(predictions, labels, scope="mean_squared_error_value"):
        with tf.name_scope(scope):
            return tf.losses.mean_squared_error(labels, predictions)

    @staticmethod
    def error(predictions, labels, scope="mean_squared_error_delta"):
        with tf.name_scope(scope):
            return tf.subtract(predictions, labels)
