import tensorflow as tf
from tensorflow.initializers import he_normal
from layer.weight_layer.weight_layer import WeightLayer


class ConvolutionalLayer(WeightLayer):
    def __init__(self, filter_dim, num_of_filters, strides, padding, func=tf.nn.conv2d, use_cudnn_on_gpu=True, data_format='NHWC',
                 dilations=[1, 1, 1, 1], filters_initializer=tf.initializers.he_normal, add_biases=False,
                 biases_initializer=tf.zeros_initializer, scope="convolutional_layer"):
        super().__init__(scope=scope)
        self.filter_dim = filter_dim
        self.num_of_filters = num_of_filters
        self.strides = strides
        self.padding = padding
        self.func = func
        self.use_cudnn_on_gpu = use_cudnn_on_gpu
        self.data_format = data_format
        self.dilations = dilations
        self.filters_initializer = filters_initializer
        self.add_biases = add_biases
        self.biases_initializer = biases_initializer

    def __str__(self):
        return f"ConvolutionalLayer({self.filter_dim} {self.strides})"

    def build_forward(self, input, remember_input=False, gather_stats=False):
        if remember_input:
            self.save_input(input)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            filter_shape = [self.filter_dim[0], self.filter_dim[1], input.shape[3], self.num_of_filters]
            filters = tf.get_variable("filters", filter_shape,
                                      initializer=self.filters_initializer(), use_resource=True)
            self.variables.append(filters)
            output = self.func(input, filters, strides=self.strides, padding=self.padding, use_cudnn_on_gpu=self.use_cudnn_on_gpu, data_format=self.data_format, dilations=self.dilations)
            if self.add_biases:
                biases = tf.get_variable("biases", output.shape[1:], initializer=self.biases_initializer(), use_resource=True)
            return output

    def gather_stats_backward(self, gradients):
        delta_input = gradients[0]
        delta_filters = gradients[1]
        filters = self.variables[0]
        tf.summary.image(f"delta_input", put_kernels_on_grid(delta_input), 1)
        tf.summary.image(f"delta_filters", put_kernels_on_grid(delta_filters), 1)
        tf.summary.image(f"filters", put_kernels_on_grid(filters), 1)


def put_kernels_on_grid(kernel, grid_Y=None, grid_X=None, pad=1):                                         
    def factorization(n):
        from numpy.ma import sqrt
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                return i, int(n / i)
    if grid_Y is None:
        (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)
    x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad
    channels = kernel.get_shape()[2]
    x = tf.transpose(x, (3, 0, 1, 2))
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))
    x = tf.transpose(x, (0, 2, 1, 3))
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))
    x = tf.transpose(x, (2, 1, 3, 0))
    x = tf.transpose(x, (2, 0, 1, 3))
    return tf.image.convert_image_dtype(x, dtype=tf.uint8)
