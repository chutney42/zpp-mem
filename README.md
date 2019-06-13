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

## Memory tracing

We implemented possibility to trace memory usage profile durning single iteration of training.

Change flag in configuration
```python
    memory_only: True

```
Your run session will be interputted after first run and in directory `./plots/` there will be file *.png with plot and *.txt with raw data to analise.

Raw data format:
```csv
1560104632231075 215296 0 forward/DFA_fully_connected_layer_8/fa_fc/IdentityN
1560104632231086 235520 20224 forward/DFA_fully_connected_layer_8/Add
1560104632231104 235520 0 forward/DFA_sigmoid_layer_9/Sigmoid
```
- First column is timestamp in microseconds.
- Second is current memory usage.
- Third is change of usage introduced in given operation.
- Last column is name of operation which invoked memory change. Refer to computation graph created in `./demo` by setting flag:
```python
    "save_graph": True
```

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
