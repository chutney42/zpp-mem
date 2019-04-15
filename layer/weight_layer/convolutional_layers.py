import tensorflow as tf

from layer.weight_layer.weight_layer import WeightLayer


class ConvolutionalLayer(WeightLayer):
    def __init__(self, filter_dim, num_of_filters, strides=[1,1,1,1], padding="SAME", func=tf.nn.conv2d, use_cudnn_on_gpu=True, data_format='NHWC',
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
        return f"ConvolutionalLayer({self.filter_dim}, {self.num_of_filters}, {self.strides}, {self.padding})"

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
                self.variables.append(biases)
                output = tf.add(output, biases)

            if gather_stats:
                tf.summary.histogram(f"filters", filters, family=self.scope)

            return output

    def gather_stats_backward(self, gradients):
        tf.summary.histogram("delta_input", gradients[0], family=self.scope)
        tf.summary.histogram("delta_filters", gradients[1], family=self.scope)

