import tensorflow as tf
import numpy as np


def load_mnist():
    training, test = tf.keras.datasets.mnist.load_data()

    def transform(feature, label):
        feature = tf.reshape(feature, [-1])
        label = tf.one_hot(label, 10)
        feature = tf.to_float(feature) / 255.0
        return feature, label

    train_data_set = tf.data.Dataset.from_tensor_slices(training).map(transform)
    test_data_set = tf.data.Dataset.from_tensor_slices(test).map(transform)

    return train_data_set, test_data_set


def shuffle(features, labels):
    tmp = list(zip(features, labels))
    np.random.shuffle(tmp)
    return list(zip(*tmp))
