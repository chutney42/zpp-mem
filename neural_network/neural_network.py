import os
from layer.layer import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # hacked by Adam

file_name = "run_auto_increment"


class NeuralNetwork(object):
    def __init__(self, types, shapes, sequence, cost_function, optimizer, scope="main", gather_stats=False,
                 save_graph=False, restore_model=False, save_model=False, restore_model_path=None, save_model_path=None):
        print(f"Create {scope} model")
        self.scope = scope
        self.sequence = sequence
        self.optimizer = optimizer
        self.cost_function = cost_function
        self.training_mode = tf.placeholder(tf.bool)
        for i, layer in enumerate(self.sequence):
            layer.scope = f"{self.scope}_{layer.scope}_{i}"
            layer.training_mode = self.training_mode

        self.handle = tf.placeholder(tf.string, shape=[], name="handle")
        with tf.variable_scope("iterator"):
            self.iterator = tf.data.Iterator.from_string_handle(self.handle, types, tuple(
                [tf.TensorShape([None] + shape.as_list()[1:]) for shape in shapes]))
            self.features, self.labels = self.iterator.get_next()
        self.result = None
        self.gather_stats = gather_stats
        self.save_graph = save_graph
        self.__init_run_number()
        self.__init_model_saving(restore_model, save_model, restore_model_path, save_model_path)
        self.step = None
        self.build()
        self.merged_summary = tf.summary.merge_all()
        self.__print_model_metadata(shapes)

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

    def __print_model_metadata(self, shapes):
        print(f"data_{self.scope}_{self.run_number}")
        print(f"input dims: {[x.value for x in shapes[0]]} output dims: {[x.value for x in shapes[1]]}")
        for layer in self.sequence:
            print(str(layer))

    def build_forward(self):
        raise NotImplementedError("This method should be implemented in subclass")

    def build_backward(self, error):
        raise NotImplementedError("This method should be implemented in subclass")

    def build_test(self, a):
        self.acc, self.acc_update = tf.metrics.accuracy(tf.argmax(self.labels, 1), tf.argmax(a, 1),
                                                        name="accuracy_metric")
        self.loss, self.loss_update = tf.metrics.mean(self.cost, name="loss_metric")

        self.running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy_metric")
        self.running_vars += tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="loss_metric")
        self.running_vars_initializer = tf.variables_initializer(var_list=self.running_vars)
        self.acc_summary = tf.summary.scalar("accuracy", self.acc)
        self.loss_summary = tf.summary.scalar("loss", self.loss)

    def build(self):
        self.result = self.build_forward()
        self.cost = self.cost_function(self.labels, self.result)
        self.build_test(self.result)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.step = self.optimizer.minimize(self.cost)
        self.step = tf.group([self.step, update_ops])

    def train(self, training_set, validation_set, epochs=2, eval_period=1000, stat_period=100,
              memory_only=False, minimum_accuracy=[]):
        self.memory_only = memory_only
        print(f" epochs: {epochs} eval_per: {eval_period} stat_per: {stat_period}")
        training_set = training_set

        with tf.variable_scope("iterators_handlers", reuse=tf.AUTO_REUSE):
            self.training_it = training_set.make_initializable_iterator()
            self.validation_it = validation_set.make_initializable_iterator()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as self.sess:
            self.__train_all_epochs(epochs, eval_period, stat_period, minimum_accuracy)

    def __train_all_epochs(self, epochs, eval_period, stat_period, minimum_accuracy):
        def should_terminate_training(current_accuracy, minimum_accuracy):
            for min_acc in minimum_accuracy:
                if self.epoch + 1 > min_acc[0] and current_accuracy < min_acc[1]:
                    print(f"Terminating learning process due to insuficcient accuracy\n Expected {min_acc[1]}" +
                          f" accuracy after {min_acc[0]} epochs, network achieved {current_accuracy} accuracy")
                    return True
            return False
        
        writer, val_writer = self.__init_writers()
        training_handle, validation_handle = self.__init_handlers()
        self.__init_global_variables()
        self.sess.run(self.running_vars_initializer)
        self.counter = 0
        self.epoch = 0
        print(f"start training for epochs={epochs}")
        for _ in range(epochs):
            self.__train_single_epoch(self.training_it, training_handle, writer, eval_period, stat_period)
            acc, loss = self.__validate(self.validation_it, validation_handle, val_writer)
            print(f"finished epoch {self.epoch}, accuracy: {acc}%, loss:{loss}")

            if self.memory_only or should_terminate_training(acc, minimum_accuracy):
                break

        acc, loss = self.__validate(self.validation_it, validation_handle)
        print(f"total accuracy: {acc}%, loss: {loss}, acc iterations: {self.counter}")
        self.__close_writers(writer, val_writer)
        self.__maybe_save_model()

    def __init_writers(self):
        if self.save_graph:
            writer = tf.summary.FileWriter(f"./demo/{self.scope}_{self.run_number}", self.sess.graph)
            val_writer = tf.summary.FileWriter(f"./demo/val_{self.scope}_{self.run_number}", self.sess.graph)
        else:
            writer = tf.summary.FileWriter(f"./demo/{self.scope}_{self.run_number}")
            val_writer = tf.summary.FileWriter(f"./demo/val_{self.scope}_{self.run_number}")
        return writer, val_writer

    def __close_writers(self, writer, val_writer):
        if self.gather_stats or self.memory_only:
            writer.close()
            val_writer.close()

    def __init_handlers(self):
        training_handle = self.sess.run(self.training_it.string_handle())
        validation_handle = self.sess.run(self.validation_it.string_handle())

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

    def __train_single_epoch(self, training_it, training_handle, writer, eval_period, stat_period):
        self.sess.run(training_it.initializer)
        while True:
            try:
                feed_dict = {self.handle: training_handle, self.training_mode: True}

                if self.memory_only or (self.gather_stats and self.counter % stat_period is 0):
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _, acc, loss = self.sess.run(
                        [self.merged_summary, self.step, self.acc_update, self.loss_update], feed_dict, run_options,
                        run_metadata)
                    writer.add_run_metadata(run_metadata, f"step_{self.counter}")
                    writer.add_summary(summary, self.counter)

                    if self.memory_only or self.counter is stat_period:
                        self.__gather_memory_usage(run_metadata)
                        if self.memory_only:
                            break
                else:
                    summary, loss_summary, _, acc, loss = self.sess.run(
                        [self.acc_summary, self.loss_summary, self.step, self.acc_update, self.loss_update], feed_dict)
                    writer.add_summary(loss_summary, self.counter)
                    writer.add_summary(summary, self.counter)


                if self.counter % eval_period is 0:
                    print(f"iteration: {self.counter}, accuracy: {acc*100}%, loss: {loss}")

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
                feed_dict = {self.handle: validation_handle, self.training_mode: False}
                self.sess.run([self.acc_update, self.loss_update], feed_dict)
            except tf.errors.OutOfRangeError:
                break
        if writer is None:
            acc, loss = self.sess.run([self.acc, self.loss])
        else:
            summary_acc, acc, summary_loss, loss = self.sess.run(
                [self.acc_summary, self.acc, self.loss_summary, self.loss])
            writer.add_summary(summary_acc, self.counter)
            writer.add_summary(summary_loss, self.counter)
        self.__switch_context()
        return acc * 100, loss

    def __gather_memory_usage(self, run_metadata):
        print(f"gather memory stats in file ./memory_usage/data_{self.scope}_{self.run_number}")
        options = tf.profiler.ProfileOptionBuilder.time_and_memory()
        options = tf.profiler.ProfileOptionBuilder(options) \
            .with_file_output(f"./memory_usage/data_{self.scope}_{self.run_number}") \
            .select(("bytes", "peak_bytes", "output_bytes", "residual_bytes")) \
            .build()
        tf.profiler.profile(self.sess.graph, run_meta=run_metadata, cmd="scope", options=options)

    def __save_context(self):
        self.context = self.sess.run(self.running_vars)

    def __switch_context(self):
        for var, val in zip(self.running_vars, self.context):
            self.sess.run(tf.assign(var, val))

    def infer(self, x):
        # TODO?
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.restore_model_path)
            res = sess.run(self.result, feed_dict={self.features: x, self.training_mode: False})
        return res

    def test(self, data_set, batch_size=10):
        # TODO?
        next_batch = data_set.batch(batch_size).make_one_shot_iterator().get_next()
        total_res = 0
        counter = 0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.restore_model_path)
            while True:
                try:
                    batch_xs, batch_ys = sess.run(next_batch)
                    res = sess.run(self.acc, feed_dict={self.features: batch_xs, self.labels: batch_ys, self.trainig_mode: False})
                    total_res += res
                    counter += batch_size
                except tf.errors.OutOfRangeError:
                    break
        return total_res / counter * 100
