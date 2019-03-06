import numpy as np
import tensorflow as tf

def tf_sigmoid_prime(x):
    return tf.multiply(tf.sigmoid(x), (tf.constant(1.0) - tf.sigmoid(x)))
