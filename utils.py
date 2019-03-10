import tensorflow as tf

def sigmoid_prime(x, name=None):
    with tf.name_scope(name or "sigmoid_prime"):
        return tf.multiply(tf.sigmoid(x), (tf.constant(1.0) - tf.sigmoid(x)))
