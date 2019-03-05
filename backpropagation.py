import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from utils import *
from layer import *
from loader import *


incr_file_name = "run_auto_increment"


class NeuralNetwork(object):
    def __init__(self, input_dim, sequence, output_dim, learning_rate=0.1, scope="main", restore_model=False,
                 save_model=False):
        self.scope = scope
        self.sequence = sequence
        self.learning_rate = tf.constant(learning_rate)
        self.features = tf.placeholder(tf.float32, [None, input_dim])
        self.labels = tf.placeholder(tf.float32, [None, output_dim])
        self.result = None
        self.__init_run_number()
        self.restored_model_path = './saved_model_{}_{}/model.ckpt'.format(self.scope, self.run_number - 1)
        self.saved_model_path = './saved_model_{}_{}/model.ckpt'.format(self.scope, self.run_number)
        self.restore_model = restore_model
        self.save_model = save_model

    def __init_run_number(self):
        if not os.path.isfile(incr_file_name):
            with open(incr_file_name, 'w+') as file:
                file.write(str(0))
        with open(incr_file_name) as file:
            self.run_number = int(file.read())
        self.run_number += 1
        with open(incr_file_name, 'w') as file:
            file.write(str(self.run_number))

    def build(self):
        self.result = self.build_forward()
        self.build_test(self.result)
        self.build_backward(self.result)

    def build_forward(self):
        a = self.features
        for i, layer in enumerate(self.sequence):
            layer.scope = '{}_{}_{}'.format(self.scope, layer.scope, i)
            a = layer.build_forward(a)
        return a

    def build_test(self, a):
        self.acct_mat = tf.equal(tf.argmax(a, 1), tf.argmax(self.labels, 1))
        self.acct_res = tf.reduce_sum(tf.cast(self.acct_mat, tf.float32))
        tf.summary.scalar("result", self.acct_res)

    def build_backward(self, output_vec):
        error = tf.subtract(output_vec, self.labels)
        self.step = []
        for i, layer in reversed(list(enumerate(self.sequence))):
            error = layer.build_backward(error)
            if layer.trainable:
                self.step.append(layer.step)

    def train(self, training_set, validation_set, batch_size=10, epoch=2, eval_period=1000):
        training_set = training_set.shuffle(200).batch(batch_size)
        iterator = tf.data.Iterator.from_structure(training_set.output_types, training_set.output_shapes)
        train_init = iterator.make_initializer(training_set)
        next_batch = iterator.get_next()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./demo/{}_{}'.format(self.scope, self.run_number), sess.graph)
            if self.restore_model:
                saver.restore(sess, self.restored_model_path)
                print("model restored from path {}".format(self.restored_model_path))
            else:
                sess.run(tf.global_variables_initializer())
                print("fresh model")
            counter = 0
            for e in range(epoch):
                res = self.validate(validation_set, sess)
                print("start epoch: {}, accuracy: {}%".format(e, res))
                sess.run(train_init)
                while True:
                    try:
                        batch_xs, batch_ys = sess.run(next_batch)
                        sess.run(self.step, feed_dict={self.features: batch_xs, self.labels: batch_ys})

                        if eval_period > 0 and counter % eval_period is 0:
                            print("iteration: {}, accuracy: {}%".format(counter, self.validate(
                                validation_set.take(1000), sess, writer, counter)))
                        counter += 1
                    except tf.errors.OutOfRangeError:
                        break
                res = self.validate(validation_set, sess)
                print("end epoch: {}, accuracy: {}%".format(e, res))

            res = self.validate(validation_set, sess)
            print("total accuracy: {}%".format(res))
            writer.close()
            if self.save_model:
                saver.save(sess, self.saved_model_path)
                print("model saved in path {}".format(self.saved_model_path))

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
                    writer.add_run_metadata(run_metadata, "step_{}".format(step))

                total_res += res
                counter += len(batch_xs)
            except tf.errors.OutOfRangeError:
                break

        return total_res / counter * 100

    def infer(self, x):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.restored_model_path)
            res = sess.run(self.result, feed_dict={self.features: x})
        return res

    def test(self, data_set, batch_size=10):
        next_batch = data_set.batch(batch_size).make_one_shot_iterator().get_next()
        total_res = 0
        counter = 0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.restored_model_path)
            while True:
                try:
                    batch_xs, batch_ys = sess.run(next_batch)
                    res = sess.run(self.acct_res, feed_dict={self.features: batch_xs, self.labels: batch_ys})
                    total_res += res
                    counter += batch_size
                except tf.errors.OutOfRangeError:
                    break
        return total_res / counter * 100


if __name__ == '__main__':
    training, test = load_mnist()
    NN = NeuralNetwork(784,
                       [FullyConnected(50),
                        BatchNormalization(),
                        Sigmoid(),
                        FullyConnected(30),
                        BatchNormalization(),
                        Sigmoid(),
                        FullyConnected(10),
                        Sigmoid()],
                       10,
                       0.1,
                       'BP')
    NN.build()
    NN.train(training, test)
