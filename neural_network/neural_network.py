import os
from layer.layer import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # hacked by Adam

file_name = "run_auto_increment"


class DirectFeedbackAlignment(object):
    def __init__(self, types, shapes, sequence, propagator, learning_rate=0.1, scope="main", gather_stats=True,
                 restore_model=False, save_model=False, restore_model_path=None, save_model_path=None):
        print(f"Create {scope} model with learning_rate={learning_rate}")
        self.scope = scope
        self.sequence = sequence
        self.propagator = propagator
        for i, block in enumerate(self.sequence):
            block.head.propagator = self.propagator
            for j, layer in enumerate(block):
                layer.scope = f"{self.scope}_{layer.scope}_{i}_{j}"
        self.learning_rate = tf.constant(learning_rate)

        self.handle = tf.placeholder(tf.string, shape=[], name="handle")
        with tf.variable_scope("iterator"):
            self.iterator = tf.data.Iterator.from_string_handle(self.handle, types, tuple(
                [tf.TensorShape([None] + shape.as_list()) for shape in shapes]))
            self.features, self.labels = self.iterator.get_next()
        self.result = None
        self.gather_stats = gather_stats
        self.__init_run_number()
        self.__init_model_saving(restore_model, save_model, restore_model_path, save_model_path)
        self.step = None
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

    def build_forward(self):
        raise NotImplementedError("This method should be implemented in subclass")

    def build_backward(self, output_vec):
        raise NotImplementedError("This method should be implemented in subclass")

    def __build_test(self, a):
        self.acc, self.acc_update = tf.metrics.accuracy(tf.argmax(self.labels, 1), tf.argmax(a, 1), name="accuracy_metric")
        self.running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy_metric")
        self.running_vars_initializer = tf.variables_initializer(var_list=self.running_vars)
        self.acc_summary = tf.summary.scalar("accuracy", self.acc)

    def build(self):
        self.result = self.build_forward()
        self.__build_test(self.result)
        self.build_backward(self.result)

    def train(self, training_set, validation_set, batch_size=20, epochs=2, eval_period=1000, stat_period=100):
        training_set = training_set.shuffle(200).batch(batch_size)

        with tf.variable_scope("itarators", reuse=tf.AUTO_REUSE):
            training_it = training_set.make_initializable_iterator()
            validation_it = validation_set.batch(batch_size).make_initializable_iterator()

        with tf.Session() as self.sess:
            self.__train_all_epochs(training_it, validation_it, batch_size, epochs, eval_period, stat_period)

    def __train_all_epochs(self, training_it, validation_it, batch_size, epochs, eval_period, stat_period):
        writer, val_writer = self.__init_writers()
        training_handle, validation_handle = self.__init_handlers(training_it, validation_it)
        self.__init_global_variables()
        self.sess.run(self.running_vars_initializer)
        self.counter = 0
        self.epoch = 0
        print(f"start training for epochs={epochs} with batch_size={batch_size}")
        for _ in range(epochs):
            res = self.__validate(validation_it, validation_handle)
            print(f"start epoch {self.epoch}, accuracy: {res}%")
            self.__train_single_epoch(training_it, validation_it, training_handle, validation_handle, writer,
                                      val_writer, eval_period, stat_period)

        res = self.__validate(validation_it, validation_handle)
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
            saver.save(self.sess, self.save_model_path, write_meta_graph=False)
            print(f"model saved in path {self.save_model_path}")

    def __train_single_epoch(self, training_it, validation_it, training_handle, validation_handle, writer,
                             val_writer, eval_period, stat_period):
        self.sess.run(training_it.initializer)
        while True:
            try:
                feed_dict = {self.handle: training_handle}

                if self.gather_stats and self.counter % stat_period is 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _, _ = self.sess.run([self.merged_summary, self.step, self.acc_update], feed_dict, run_options, run_metadata)
                    writer.add_run_metadata(run_metadata, f"step_{self.counter}")
                    writer.add_summary(summary, self.counter)
                else:
                    self.sess.run([self.step, self.acc_update], feed_dict)

                if self.counter % eval_period is 0:
                    res = self.__validate(validation_it, validation_handle, val_writer)
                    print(f"iteration: {self.counter}, accuracy: {res}%")

                self.counter += 1
            except tf.errors.OutOfRangeError:
                break
        self.epoch += 1

    def __validate(self, validation_iterator, validation_handle, writer=None):
        self.sess.run(validation_iterator.initializer)
        self.__save_context()
        self.sess.run(self.running_vars_initializer)
        while True:
            try:
                feed_dict = {self.handle: validation_handle}
                self.sess.run(self.acc_update, feed_dict)
            except tf.errors.OutOfRangeError:
                break
        if writer is None:
            res = self.sess.run(self.acc)
        else:
            summary, res = self.sess.run([self.acc_summary, self.acc])
            writer.add_summary(summary, self.counter)
        self.__switch_context()
        return res * 100

    def __save_context(self):
        self.context = self.sess.run(self.running_vars)

    def __switch_context(self):
        for var, val in zip(self.running_vars, self.context):
            self.sess.run(tf.assign(var, val))

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
                    res = sess.run(self.acc, feed_dict={self.features: batch_xs, self.labels: batch_ys})
                    total_res += res
                    counter += batch_size
                except tf.errors.OutOfRangeError:
                    break
        return total_res / counter * 100
