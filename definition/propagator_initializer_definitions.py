import tensorflow as tf


def normal():
    return tf.initializers.random_normal()


def he_normal():
    return tf.initializers.he_normal()


def uniform():
    return tf.initializers.random_uniform()


def uniform1():
    return tf.initializers.random_uniform(minval=-1.0, maxval=1.0)


def he_uniform():
    return tf.initializers.he_uniform()


def lecun_uniform():
    return tf.initializers.lecun_uniform()


def lecun_normal():
    return tf.initializers.lecun_normal()
