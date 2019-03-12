import tensorflow as tf


def load(dataset_name):
    if dataset_name == 'mnist':
        return load_mnist()
    elif dataset_name == 'cifar10':
        return load_cifar10()
    elif dataset_name == 'cifar100':
        return load_cifar100()
    else:
        return

def load_cifar100():
    training, test = tf.keras.datasets.cifar100.load_data()

    def transform(feature, label):
        feature = tf.reshape(feature, [-1])
        label = label[0]
        label = tf.one_hot(label, 100)
        feature = tf.to_float(feature) / 255.0
        return feature, label

    train_data_set = tf.data.Dataset.from_tensor_slices(training).map(transform)
    test_data_set = tf.data.Dataset.from_tensor_slices(test).map(transform)

    return train_data_set, test_data_set


def load_cifar10():
    training, test = tf.keras.datasets.cifar10.load_data()

    def transform(feature, label):
        feature = tf.reshape(feature, [-1])
        label = label[0]
        label = tf.one_hot(label, 10)
        feature = tf.to_float(feature) / 255.0
        return feature, label

    train_data_set = tf.data.Dataset.from_tensor_slices(training).map(transform)
    test_data_set = tf.data.Dataset.from_tensor_slices(test).map(transform)

    return train_data_set, test_data_set

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
