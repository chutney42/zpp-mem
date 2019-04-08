import tensorflow as tf
import numpy as np
import os


def cifar100(batch_size=16):
    filenames = ["./datasets/cifar100_train.tfrecords", "./datasets/cifar100_test.tfrecords"]

    found = True

    for file in filenames:
        if not os.path.isfile(file):
            found = False
            print("TFRecords not found for file: " + file)

    if not found:
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

        train_x = train_x.astype(np.float32)
        test_x = test_x.astype(np.float32)

        n_mean = np.mean(train_x)
        n_max = np.max(train_x)
        n_min = np.min(train_x)
        train_x = (train_x - n_mean) / (n_max - n_min)
        test_x = (test_x - n_mean) / (n_max - n_min)

        create_tf_record(examples=train_x, labels=train_y, path=filenames[0])
        create_tf_record(examples=test_x, labels=test_y, path=filenames[1])

    train = tf.data.TFRecordDataset(filenames[0])
    test = tf.data.TFRecordDataset(filenames[1])

    train = train.shuffle(10000, seed=10)
    train = train.apply(
        tf.contrib.data.map_and_batch(parse_example, batch_size=batch_size, num_parallel_batches=3)
        )
    train = train.prefetch(100)

    test = test.shuffle(10000, seed=10)
    test = test.apply(
        tf.contrib.data.map_and_batch(parse_example, batch_size=batch_size, num_parallel_batches=3)
        )
    test = test.prefetch(100)

    return train, test


def cifar10(batch_size=16):
    training, test = tf.keras.datasets.cifar10.load_data()

    def transform(feature, label):
        feature = tf.reshape(feature, [32, 32, 3])
        label = label[0]
        label = tf.one_hot(label, 10)
        feature = (tf.to_float(feature) - 128.0) / 255.0
        return feature, label

    train_data_set = tf.data.Dataset.from_tensor_slices(training).map(transform)
    test_data_set = tf.data.Dataset.from_tensor_slices(test).map(transform)

    return train_data_set, test_data_set


def mnist(batch_size=16):
    filenames = ["mnist_train.tfrecords", "mnist_test.tfrecords"]
    found = True

    for file in filenames:
        if not os.path.isfile('./datasets/' + file):
            found = False
            print("TFRecords not found for file: " + file)

    training, test = tf.keras.datasets.mnist.load_data()

    def transform(feature, label):
        feature = tf.reshape(feature, [28, 28, 1])
        label = tf.one_hot(label, 10)
        feature = (tf.to_float(feature) - 128.0) / 255.0
        return feature, label

    train_data_set = tf.data.Dataset.from_tensor_slices(training).map(transform)
    test_data_set = tf.data.Dataset.from_tensor_slices(test).map(transform)

    return train_data_set, test_data_set


def create_tf_record(examples, labels, path):
    # Takes training examples and labels to save in a .tfrecord file at the given path

    with tf.python_io.TFRecordWriter(path) as writer:
        # Make examples into serialized string
        # we can just save all images

        # Loop through all images
        for i in range(examples.shape[0]):
            # turn image into bytes and get our label
            img = examples[i].tostring()
            label = labels[i]

            # Now we need an example which is made up of a feature

            features = tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }
            )

            example = tf.train.Example(features=features)

            serialized = example.SerializeToString()

            writer.write(serialized)

            # Update the progressbar


def parse_example(serialized):
    features = {'image': (tf.FixedLenFeature((), tf.string, default_value="")),
                'label': (tf.FixedLenFeature((), tf.int64, default_value=0))}

    parsed = tf.parse_single_example(serialized=serialized, features=features)

    raw_image = parsed['image']

    image = tf.decode_raw(raw_image, tf.float32)

    label = tf.one_hot(parsed['label'], 100)

    return tf.reshape(image, [32, 32, 3]), label
