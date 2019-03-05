import os
import tensorflow as tf
from utils import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

file_name = "run_auto_increment"


class NeuralNetwork(object):
    def __init__(self, input_dim, sequence, output_dim, learning_rate=0.1, scope="main"):
        self.scope = scope
        self.sequence = sequence
        self.learning_rate = tf.constant(learning_rate)
        self.features = tf.placeholder(tf.float32, [None, input_dim])
        self.labels = tf.placeholder(tf.float32, [None, output_dim])
        self.result = None
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

    def train(self, training_set, validation_set, batch_size=10, epoch=2, eval_period=1000):
        training_set = training_set.shuffle(200).batch(batch_size)
        iterator = tf.data.Iterator.from_structure(training_set.output_types, training_set.output_shapes)
        train_init = iterator.make_initializer(training_set)
        next_batch = iterator.get_next()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("./demo/{}_{}".format(self.scope, self.run_number), sess.graph)
            sess.run(tf.global_variables_initializer())
            counter = 0
            for e in range(epoch):
                sess.run(train_init)
                while True:
                    try:
                        batch_xs, batch_ys = sess.run(next_batch)
                        sess.run(self.step, feed_dict={self.features: batch_xs, self.labels: batch_ys})

                        if eval_period > 0 and counter % eval_period is 0:
                            print("iter: {}, acc: {}%".format(counter,
                                                              self.validate(validation_set.take(1000), sess, writer,
                                                                            counter)))

                        counter += 1
                    except tf.errors.OutOfRangeError:
                        break
                res = self.validate(validation_set, sess)
                print("epoch {}:  {}%".format(e, res))

            res = self.validate(validation_set, sess)
            print("total {}%".format(res))
            writer.close()
            # TODO save model

    def validate(self, validation_set, sess, writer=None, step=0):
        total_res = 0
        counter = 0
        # hacky way to have only one batch
        next_batch = validation_set.batch(10000000).make_one_shot_iterator().get_next()
        while True:
            try:
                batch_xs, batch_ys = sess.run(next_batch)
                if writer is None:
                    res = sess.run(self.acct_res, feed_dict={self.features: batch_xs, self.labels: batch_ys})
                else:
                    merged = tf.summary.merge_all()
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, res = sess.run([merged, self.acct_res], options=run_options, run_metadata=run_metadata,
                                            feed_dict={self.features: batch_xs, self.labels: batch_ys})
                    writer.add_summary(summary, step)
                    writer.add_run_metadata(run_metadata, 'step%d' % step)

                total_res += res
                counter += len(batch_xs)
            except tf.errors.OutOfRangeError:
                break

        return total_res / counter * 100

    def infer(self, x):
        #TODO restore model
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
