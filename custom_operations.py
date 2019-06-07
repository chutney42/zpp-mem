import tensorflow as tf
import numpy as np

def feedback_alignment_fc(input, weights, initializer=tf.initializers.he_normal(), name="fa_fc"):
    random = tf.get_variable("random", shape=reversed(weights.get_shape().as_list()),
                             initializer=initializer, use_resource=True, trainable=False)
    @tf.custom_gradient
    def func(x):
        def grad(dy, variables=[weights]):
            dx = tf.matmul(dy, random)
            dw = tf.matmul(tf.transpose(x), dy)
            return dx, [dw]
        return tf.matmul(x, weights), grad
    with tf.name_scope(name):
        return func(input)

def feedback_alignment_conv(input, weights, strides, padding, use_cudnn_on_gpu=True, data_format='NHWC',
                            dilations=[1, 1, 1, 1], initializer=tf.initializers.he_normal(),
                            name="fa_conv"):
    random = tf.get_variable("random", shape=weights.get_shape().as_list(), initializer=initializer, use_resource=True, trainable=False)
    @tf.custom_gradient
    def func(x):
        def grad(dy, variables=[weights]):
            dx = tf.nn.conv2d_backprop_input(tf.shape(x), random, dy, strides, padding, use_cudnn_on_gpu,
                                             data_format, dilations)
            dw = tf.nn.conv2d_backprop_filter(x, weights.get_shape(), dy, strides, padding, use_cudnn_on_gpu,
                                             data_format, dilations)
            return dx, [dw]
        return tf.nn.conv2d(input, weights, strides, padding, use_cudnn_on_gpu, data_format, dilations), grad
    with tf.name_scope(name):
        return func(input)

def direct_feedback_alignment_fc(input, weights, output_dim, error_container, initializer=tf.initializers.he_normal(),
                                 name="fa_fc"):
    random = tf.get_variable("random", shape=[output_dim, weights.shape[0]], initializer=initializer, use_resource=True, trainable=False)
    @tf.custom_gradient
    def func(x):
        def grad(dy, variables=[weights]):
            dx = tf.matmul(error_container[0], random, name='matmul_grad_x')
            dw = tf.matmul(tf.transpose(x), dy, name='matmul_grad_w')
            return dx, [dw]
        return tf.matmul(x, weights, name='matmul_forward_x'), grad
    with tf.name_scope(name):
        return func(input)
       
def direct_feedback_alignment_conv(input, weights, output_dim, error_container, strides, padding,
                                   use_cudnn_on_gpu=True, data_format='NHWC', dilations=[1, 1, 1, 1],
                                   initializer=tf.initializers.he_normal(), name="fa_conv"):
    input_shape = tf.shape(input)
    input_flat_shape = np.prod(input.shape[1:])
    random = tf.get_variable("random", shape=[output_dim, input_flat_shape],
                             initializer=initializer, use_resource=True, trainable=False)
    @tf.custom_gradient
    def func(x):
        def grad(dy, variables=[weights]):
            dx = tf.reshape(tf.matmul(error_container[0], random), input_shape)
            dw = tf.nn.conv2d_backprop_filter(x, weights.get_shape(), dy, strides, padding, use_cudnn_on_gpu,
                                             data_format, dilations)
            return dx, [dw]
        return tf.nn.conv2d(input, weights, strides, padding, use_cudnn_on_gpu, data_format, dilations), grad
    with tf.name_scope(name):
        return func(input)

