import tensorflow as tf


def cifar100():
    training, test = tf.keras.datasets.cifar100.load_data()

    def transform(feature, label):
        feature = tf.reshape(feature, [32,32,3])
        label = label[0]
        label = tf.one_hot(label, 100)
        feature = (tf.to_float(feature) - 128.0 / 255.0)
        return feature, label

    train_data_set = tf.data.Dataset.from_tensor_slices(training).map(transform)
    test_data_set = tf.data.Dataset.from_tensor_slices(test).map(transform)

    return train_data_set, test_data_set


def cifar10():
    training, test = tf.keras.datasets.cifar10.load_data()

    def transform(feature, label):
        feature = tf.reshape(feature, [32,32,3])
        label = label[0]
        label = tf.one_hot(label, 10)
        feature = tf.to_float(feature) - 128.0 / 255.0
        return feature, label

    train_data_set = tf.data.Dataset.from_tensor_slices(training).map(transform)
    test_data_set = tf.data.Dataset.from_tensor_slices(test).map(transform)

    return train_data_set, test_data_set


def mnist():
    training, test = tf.keras.datasets.mnist.load_data()

    def transform(feature, label):
        feature = tf.reshape(feature, [28, 28, 1])
        label = tf.one_hot(label, 10)
        feature = tf.to_float(feature) - 128.0 / 255.0
        return feature, label

    train_data_set = tf.data.Dataset.from_tensor_slices(training).map(transform)
    test_data_set = tf.data.Dataset.from_tensor_slices(test).map(transform)

    return train_data_set, test_data_set
