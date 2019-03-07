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

    def train(self, training_set, validation_set, batch_size=20, epoch=2, eval_period=1000, stat_period=100):
        training_set = training_set.shuffle(200).batch(batch_size)
        iterator = tf.data.Iterator.from_string_handle(self.handle, training_set.output_types,
                                                       training_set.output_shapes)

        training_iterator = training_set.make_initializable_iterator()
        # hacky way to have only one batch
        validation_iterator = validation_set.batch(100000).make_initializable_iterator()

        next_batch = iterator.get_next()

        with tf.Session() as sess:

            writer = tf.summary.FileWriter("./demo/{}_{}".format(self.scope, self.run_number), sess.graph)
            val_writer = tf.summary.FileWriter("./demo/val_{}_{}".format(self.scope, self.run_number), sess.graph)

            training_handle = sess.run(training_iterator.string_handle())
            validation_handle = sess.run(validation_iterator.string_handle())

            sess.run(tf.global_variables_initializer())
            merged = tf.summary.merge_all()
            counter = 0
            print(f"Start training for epochs={epoch} with batch_size={batch_size}")
            for e in range(epoch):
                sess.run(training_iterator.initializer)
                while True:
                    try:
                        batch_xs, batch_ys = sess.run(next_batch, feed_dict={self.handle: training_handle})

                        if self.gather_stats and counter % stat_period is 0:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, _ = sess.run([merged, self.step], options=run_options, run_metadata=run_metadata,
                                                  feed_dict={self.features: batch_xs, self.labels: batch_ys})
                            writer.add_run_metadata(run_metadata, 'step_%d' % counter)
                        else:
                            summary, _ = sess.run([merged, self.step],
                                                  feed_dict={self.features: batch_xs, self.labels: batch_ys})

                        writer.add_summary(summary, counter)

                        if eval_period > 0 and counter % eval_period is 0:
                            print("iter: {}, acc: {}%".format(counter, self.__validate(next_batch, validation_iterator,
                                                                                     validation_handle, sess,
                                                                                     val_writer, counter)))

                        counter += 1
                    except tf.errors.OutOfRangeError:
                        break
                res = self.__validate(next_batch, validation_iterator, validation_handle, sess)
                print("epoch {}:  {}%".format(e, res))

            res = self.__validate(next_batch, validation_iterator, validation_handle, sess)
            print("total {}%".format(res))
            writer.close()
            val_writer.close()
            # TODO save model

    def __validate(self, get_next_op, validation_iterator, validation_handle, sess, writer=None, step=0):
        total_res = 0
        counter = 0
        merged = tf.summary.merge_all()
        sess.run(validation_iterator.initializer)
        while True:
            try:
                batch_xs, batch_ys = sess.run(get_next_op, feed_dict={self.handle: validation_handle})
                feed_dict = {self.features: batch_xs, self.labels: batch_ys}
                total_res += self.__validate_single_batch(sess, feed_dict, merged, writer, step)
                counter += len(batch_xs)
            except tf.errors.OutOfRangeError:
                break

        return total_res / counter * 100

    def __validate_single_batch(self, sess, feed_dict, merged, writer=None, step=0):
        if writer is None:
            res = sess.run(self.acct_res, feed_dict=feed_dict)
        else:
            summary, res = sess.run([merged, self.acct_res], feed_dict=feed_dict)
            writer.add_summary(summary, step)
        return res

    def infer(self, x):
        # TODO restore model
        with tf.Session() as sess:
            res = sess.run(self.result, feed_dict={self.features: x})
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
                    res = sess.run(self.acct_res, feed_dict={self.features: batch_xs, self.labels: batch_ys})
                    total_res += res
                    counter += batch_size
                except tf.errors.OutOfRangeError:
                    break
        return total_res / counter * 100
