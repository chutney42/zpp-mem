# ZPP-MEM

## [TensorBoard](https://github.com/tensorflow/tensorboard)
### Usage
```python
def backpropagation(self, ...):
    tf.summary.scalar("scalar_name", scalar)
    tf.summary.histogram("histogram_name", tensor)
    tf.summary.image("image_name", image)
    # example of image: tf.reshape(weights, (1, weights.shape[0], weights.shape[1], 1))
    ...

def train(self, ...):
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("logs/log_name", sess.graph)
        ...
        for i in range(...):
            ...
            merged = tf.summary.merge_all()
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, self.step], options=run_options, run_metadata=run_metadata, ...)
            writer.add_summary(summary, i)
            writer.add_run_metadata(run_metadata, 'step%d' % i)
        writer.close()
```

### Running
`$ tensorboard --logdir=logs`

## [TensorFlow Profiler](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler)
### Installation
First install bazel from [here](https://docs.bazel.build/versions/master/bazel-overview.html).
```
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow/
$ bazel build tensorflow/core/profiler:profiler
```

### Code
```python
def train(self, ...):
    builder = tf.profiler.ProfileOptionBuilder
    opts = builder(builder.time_and_memory()).order_by('micros').with_empty_output().build()
    with tf.contrib.tfprof.ProfileContext('profile_out/', trace_steps=[], dump_steps=[]) as pctx:
        with tf.Session() as sess:
            ...
            for i in range(...):
                ...
                pctx.trace_next_step()
                pctx.dump_next_step()
                sess.run(...)
                pctx.profiler.profile_operations(options=opts)
```

### Usage
```
$ cd tensorflow/
$ bazel-bin/tensorflow/core/profiler/profiler --profile_path=profile_out/profile_xxx
tfprof> graph -step -1 -max_depth 100000 -output timeline:outfile=<filename>
```
Then open this URL: chrome://tracing in Chrome and load timeline file.
