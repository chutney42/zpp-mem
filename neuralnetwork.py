import os
import tensorflow as tf
from utils import *
from layer import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

file_name = "run_auto_increment"


class NeuralNetwork(object):
    def __init__(self, input_dim, sequence, output_dim, learning_rate=0.1, scope="main", gather_stats=True,
                 restore_model=False, save_model=False, restore_model_path=None, save_model_path=None):
        self.scope = scope
        self.sequence = sequence
        self.learning_rate = tf.constant(learning_rate)
        self.features = tf.placeholder(tf.float32, [None, input_dim])
        self.labels = tf.placeholder(tf.float32, [None, output_dim])
        self.result = None
        self.gather_stats = gather_stats
        self.handle = tf.placeholder(tf.string, shape=[])
        self.__init_run_number()
        self.__init_model_saving(restore_model, save_model, restore_model_path, save_model_path)
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

    def __init_model_saving(self, restore_model, save_model, restore_model_path, save_model_path):
        self.restore_model = restore_model
        self.save_model = save_model
        if restore_model_path is not None:
            self.restore_model_path = restore_model_path
        else:
            self.restore_model_path = f"./saved_model_{self.scope}_{self.run_number - 1}/model.ckpt"
        if save_model_path is not None:
            self.save_model_path = save_model_path
        else:
            self.save_model_path = f"./saved_model_{self.scope}_{self.run_number}/model.ckpt"

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
        self.__init_global_variables()
        self.counter = 0
        self.epoch = 0
        print(f"start training for epochs={epochs} with batch_size={batch_size}")
        for _ in range(epochs):
            res = self.__validate(batch, validation_it, validation_handle)
            print(f"start epoch {self.epoch}, accuracy: {res}%")
            self.__train_single_epoch(training_it, validation_it, training_handle, validation_handle, batch, writer,
                                      val_writer, eval_period, stat_period)

        res = self.__validate(batch, validation_it, validation_handle)
        print(f"total accuracy: {res}%")
        self.__close_writers(writer, val_writer)
        self.__maybe_save_model()

    def __init_writers(self):
        writer = tf.summary.FileWriter(f"./demo/{self.scope}_{self.run_number}", self.sess.graph)
        val_writer = tf.summary.FileWriter(f"./demo/val_{self.scope}_{self.run_number}", self.sess.graph)
        return writer, val_writer

    def __close_writers(self, writer, val_writer):
        writer.close()
        val_writer.close()

    def __init_handlers(self, training_it, validation_it):
        training_handle = self.sess.run(training_it.string_handle())
        validation_handle = self.sess.run(validation_it.string_handle())
        return training_handle, validation_handle

    def __init_global_variables(self):
        if self.restore_model:
            saver = tf.train.Saver()
            saver.restore(self.sess, self.restore_model_path)
            print(f"model restored from path {self.restore_model_path}")
        else:
            self.sess.run(tf.global_variables_initializer())
            print("fresh model")

    def __maybe_save_model(self):
        if self.save_model:
            saver = tf.train.Saver()
            saver.save(self.sess, self.save_model_path)
            print(f"model saved in path {self.save_model_path}")

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
                    writer.add_run_metadata(run_metadata, f"step_{self.counter}")
                else:
                    summary, _ = self.sess.run([self.merged_summary, self.step], feed_dict)

                writer.add_summary(summary, self.counter)

                if self.counter % eval_period is 0:
                    res = self.__validate(batch, validation_it, validation_handle, val_writer)
                    print(f"iteration: {self.counter}, accuracy: {res}%")

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
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.restore_model_path)
            res = sess.run(self.result, feed_dict={self.features: x})
        return res

    def test(self, data_set, batch_size=10):
        next_batch = data_set.batch(batch_size).make_one_shot_iterator().get_next()
        total_res = 0
        counter = 0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.restore_model_path)
            while True:
                try:
                    batch_xs, batch_ys = sess.run(next_batch)
                    res = sess.run(self.acct_res, feed_dict={self.features: batch_xs, self.labels: batch_ys})
                    total_res += res
                    counter += batch_size
                except tf.errors.OutOfRangeError:
                    break
        return total_res / counter * 100
