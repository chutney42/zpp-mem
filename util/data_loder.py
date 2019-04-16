import os

import tensorflow as tf
import numpy as np


class DataLoader:
    def __init__(self, dataset_info, alternate_source=None, batch_size=16):
        self.batch_size = batch_size
        self.dataset_info = dataset_info
        self._load_data(alternate_source)

    def _load_data(self, alternate_source):
        found = True
        for file in self.dataset_info.filenames:
            if not os.path.isfile(file):
                found = False
                print("TFRecords not found for file: " + file)
        if not found:
            (train_x, train_y), (test_x, test_y) = alternate_source()

            train_x = train_x.astype(np.float32)
            test_x = test_x.astype(np.float32)

            n_mean = np.mean(train_x)
            n_max = np.max(train_x)
            n_min = np.min(train_x)
            train_x = (train_x - n_mean) / (n_max - n_min)
            test_x = (test_x - n_mean) / (n_max - n_min)

            DataLoader._create_tf_record(images=train_x, labels=train_y, path=self.dataset_info.filenames[0])
            DataLoader._create_tf_record(images=test_x, labels=test_y, path=self.dataset_info.filenames[1])

    def get_dataset(self):
        train = tf.data.TFRecordDataset(self.dataset_info.filenames[0])
        test = tf.data.TFRecordDataset(self.dataset_info.filenames[1])

        train = train.shuffle(10000, seed=10)
        train = train.apply(
            tf.data.experimental.map_and_batch(self._parse_example, batch_size=self.batch_size, num_parallel_batches=3)
        )
        train = train.prefetch(100)

        test = test.shuffle(10000, seed=10)
        test = test.apply(
            tf.data.experimental.map_and_batch(self._parse_example, batch_size=self.batch_size, num_parallel_batches=3)
        )
        test = test.prefetch(100)

        return train, test

    @staticmethod
    def _create_tf_record(images, labels, path):
        with tf.python_io.TFRecordWriter(path) as writer:
            for i in range(images.shape[0]):
                img = images[i].tostring()
                label = labels[i]

                features = tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                )
                example = tf.train.Example(features=features)
                serialized = example.SerializeToString()
                writer.write(serialized)

    def _parse_example(self, serialized):
        features = {'image': (tf.FixedLenFeature((), tf.string, default_value="")),
                    'label': (tf.FixedLenFeature((), tf.int64, default_value=0))}

        parsed = tf.parse_single_example(serialized=serialized, features=features)
        raw_image = parsed['image']
        image = tf.decode_raw(raw_image, tf.float32)
        label = tf.one_hot(parsed['label'], self.dataset_info.num_labels)
        return tf.reshape(image, self.dataset_info.shapes), label
