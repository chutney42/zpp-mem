import tensorflow as tf


def normal():
    return tf.initializers.random_normal()


def he_normal():
    return tf.initializers.he_normal()


def uniform():
    return tf.initializers.random_uniform()
