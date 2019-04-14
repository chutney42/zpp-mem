import tensorflow as tf
from util.data_loder import DataLoader
from collections import namedtuple

dataset_info = namedtuple("dataset_info", "filenames shapes num_labels")


def cifar100(batch_size=16):
    return DataLoader(
        dataset_info(["./datasets/cifar100_train.tfrecords", "./datasets/cifar100_test.tfrecords"], (32, 32, 3), 100),
        tf.keras.datasets.cifar100.load_data, batch_size).get_dataset()


def cifar10(batch_size=16):
    return DataLoader(
        dataset_info(["./datasets/cifar10_train.tfrecords", "./datasets/cifar10_test.tfrecords"], (32, 32, 3), 10),
        tf.keras.datasets.cifar10.load_data, batch_size).get_dataset()


def mnist(batch_size=16):
    return DataLoader(
        dataset_info(["./datasets/mnist_train.tfrecords", "./datasets/mnist_test.tfrecords"], (28, 28, 1), 10),
        tf.keras.datasets.mnist.load_data, batch_size).get_dataset()
