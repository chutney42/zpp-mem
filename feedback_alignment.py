import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import backpropagation
from utils import *

class FeedbackAlignment(backpropagation.NeuralNetwork):
    def backpropagation(self, input_error, z, a, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights")
            random_weights = tf.get_variable("random_weights",
                                             shape=tf.transpose(weights).get_shape().as_list(),
                                             initializer=tf.random_normal_initializer())
            biases = tf.get_variable("biases")
            error = tf.multiply(input_error, sigmoid_prime(z))
            roc_cost_over_biases = error
            roc_cost_over_weights = tf.matmul(tf.transpose(a), error)
            output_error = tf.matmul(error, random_weights)
            return output_error, \
                tf.assign(weights,
                          tf.subtract(weights,
                                      tf.multiply(self.learning_rate,
                                                  roc_cost_over_weights))), \
                tf.assign(biases,
                          tf.subtract(biases,
                                      tf.multiply(self.learning_rate,
                                                  tf.reduce_mean(roc_cost_over_biases,
                                                                 axis=[0]))))

if __name__ == '__main__':
    FA = FeedbackAlignment([784, 100, 50, 30, 10], scope="FA")
    FA.build()
    FA.train()
