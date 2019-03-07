import os
import tensorflow as tf
from utils import *
from layer import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

file_name = "run_auto_increment"


class NeuralNetwork(object):
    def __init__(self, input_dim, sequence, output_dim, learning_rate=0.1, scope="main", gather_stats=True):
        self.scope = scope
        self.sequence = sequence
        self.learning_rate = tf.constant(learning_rate)
        self.features = tf.placeholder(tf.float32, [None, input_dim])
        self.labels = tf.placeholder(tf.float32, [None, output_dim])
        self.result = None
        self.gather_stats = gather_stats
        self.handle = tf.placeholder(tf.string, shape=[])
        self.__init_run_number()
        self.build()
        self.merged_summary = tf.summary.merge_all()

    def __init_run_number(self):
        if not os.path.isfile(file_name):
            with open(file_name, 'w+') as file:
                file.write(str(0))
        with open(file_name) as file:
            self.run_number = int(file.read())
        self.run_number += 1
        with open(file_name, 'w') as file:
            file.write(str(self.run_number))

    def build(self):
        raise NotImplementedError("This method should be implemented in subclass")

    def train(self, training_set, validation_set, batch_size=20, epochs=2, eval_period=1000, stat_period=100):
        training_set = training_set.shuffle(200).batch(batch_size)
        training_it = training_set.make_initializable_iterator()
        # hacky way to have only one batch
        validation_it = validation_set.batch(100000).make_initializable_iterator()

        batch = tf.data.Iterator\
            .from_string_handle(self.handle, training_set.output_types, training_set.output_shapes)\
            .get_next()
        with tf.Session() as self.sess:
            self.__train_all_epochs(training_it, validation_it, batch, batch_size, epochs, eval_period, stat_period)

    def __train_all_epochs(self, training_it, validation_it, batch, batch_size, epochs, eval_period, stat_period):
        writer, val_writer = self.__init_writers()
        training_handle, validation_handle = self.__init_handlers(training_it, validation_it)
        self.sess.run(tf.global_variables_initializer())
        self.counter = 0
        self.epoch = 0
        print(f"Start training for epochs={epochs} with batch_size={batch_size}")
        for _ in range(epochs):
            self.__train_single_epoch(training_it, validation_it, training_handle, validation_handle, batch, writer,
                                      val_writer, eval_period, stat_period)
            res = self.__validate(batch, validation_it, validation_handle)
            print("epoch {}:  {}%".format(self.epoch, res))

        res = self.__validate(batch, validation_it, validation_handle)
        print("total {}%".format(res))
        self.__close_writers(writer, val_writer)
        # TODO save model

    def __init_writers(self):
        writer = tf.summary.FileWriter("./demo/{}_{}".format(self.scope, self.run_number), self.sess.graph)
        val_writer = tf.summary.FileWriter("./demo/val_{}_{}".format(self.scope, self.run_number), self.sess.graph)
        return writer, val_writer

    def __close_writers(self, writer, val_writer):
        writer.close()
        val_writer.close()

    def __init_handlers(self, training_it, validation_it):
        training_handle = self.sess.run(training_it.string_handle())
        validation_handle = self.sess.run(validation_it.string_handle())
        return training_handle, validation_handle

    def __train_single_epoch(self, training_it, validation_it, training_handle, validation_handle, batch, writer,
                             val_writer, eval_period, stat_period):
        self.sess.run(training_it.initializer)
        while True:
            try:
                batch_xs, batch_ys = self.sess.run(batch, {self.handle: training_handle})
                feed_dict = {self.features: batch_xs, self.labels: batch_ys}

                if self.gather_stats and self.counter % stat_period is 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = self.sess.run([self.merged_summary, self.step], feed_dict, run_options, run_metadata)
                    writer.add_run_metadata(run_metadata, 'step_%d' % self.counter)
                else:
                    summary, _ = self.sess.run([self.merged_summary, self.step], feed_dict)

                writer.add_summary(summary, self.counter)

                if self.counter % eval_period is 0:
                    res = self.__validate(batch, validation_it, validation_handle, val_writer)
                    print("iter: {}, acc: {}%".format(self.counter, res))

                self.counter += 1
            except tf.errors.OutOfRangeError:
                break
        self.epoch += 1

    def __validate(self, get_next_op, validation_iterator, validation_handle, writer=None):
        total_res = 0
        counter = 0
        merged = tf.summary.merge_all()
        self.sess.run(validation_iterator.initializer)
        while True:
            try:
                batch_xs, batch_ys = self.sess.run(get_next_op, {self.handle: validation_handle})
                feed_dict = {self.features: batch_xs, self.labels: batch_ys}
                total_res += self.__validate_single_batch(feed_dict, merged, writer)
                counter += len(batch_xs)
            except tf.errors.OutOfRangeError:
                break

        return total_res / counter * 100

    def __validate_single_batch(self, feed_dict, merged, writer=None):
        if writer is None:
            res = self.sess.run(self.acct_res, feed_dict)
        else:
            summary, res = self.sess.run([merged, self.acct_res], feed_dict)
            writer.add_summary(summary, self.counter)
        return res

    def infer(self, x):
        # TODO restore model
        with tf.Session() as sess:
            res = sess.run(self.result, {self.features: x})
        return res

    def test(self, data_set, batch_size=10):
        # TODO restore model
        next_batch = data_set.batch(batch_size).make_one_shot_iterator().get_next()
        total_res = 0
        counter = 0
        with tf.Session() as sess:
            while True:
                try:
                    batch_xs, batch_ys = sess.run(next_batch)
                    res = sess.run(self.acct_res, {self.features: batch_xs, self.labels: batch_ys})
                    total_res += res
                    counter += batch_size
                except tf.errors.OutOfRangeError:
                    break
        return total_res / counter * 100
