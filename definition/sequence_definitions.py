from definition.resnet import *
from layer import *


def fcReLu(output_size):
    return [FullyConnected(100, flatten=True), ReLu(),
            FullyConnected(30), ReLu(),
            FullyConnected(output_size)]

def fcSigmoid(output_size):
    return [FullyConnected(50, flatten=True), Sigmoid(),
            FullyConnected(30), Sigmoid(),
            FullyConnected(output_size), Sigmoid()]

def long_fc(output_size):
    sequence = []
    for i in range(30):
        sequence += [FullyConnected(500, flatten=(i == 0)), BatchNormalization(), Sigmoid()]
    sequence += [FullyConnected(output_size), Sigmoid()]
    return sequence

def long_fc2(output_size):
    sequence = []
    for i in range(20):
        sequence += [FullyConnected(500, flatten=(i == 0)), Sigmoid()]
    sequence += [FullyConnected(output_size), Sigmoid()]
    return sequence

def long_conv(output_size):
    sequence = []
    for i in range(30):
        sequence += [ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=5, strides=[1, 1], padding="SAME"), BatchNormalization(), Sigmoid()]
    sequence += [FullyConnected(output_size, flatten=True), Sigmoid()]
    return sequence

def resnet_18(output_size):
    return build_resnet(output_size, [2, 2, 2, 2], batch_relu_conv)


def resnet_34(output_size):
    return build_resnet(output_size, [3, 4, 6, 3], batch_relu_conv)


def resnet_50(output_size):
    return build_resnet(output_size, [3, 4, 6, 3], batch_relu_conv_3)


def resnet_101(output_size):
    return build_resnet(output_size, [3, 4, 23, 3], batch_relu_conv_3)


def resnet_152(output_size):
    return build_resnet(output_size, [3, 8, 36, 3], batch_relu_conv_3)


def liao_mnist(output_size):
    return [ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=20, strides=[1, 1], padding="VALID"),
            MaxPool([2, 2], [2, 2], padding="VALID"),
            ReLu(),
            ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=50, strides=[1, 1], padding="VALID"),
            MaxPool([2, 2], [2, 2], padding="VALID"),
            ReLu(),
            FullyConnected(500, flatten=True),
            ReLu(),
            FullyConnected(output_size)]

def liao_mnist_bn(output_size):
    return [ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=20, strides=[1, 1], padding="VALID"),
            MaxPool([2, 2], [2, 2], padding="VALID"),
            ReLu(),
            BatchNormalization(momentum=0.9),
            ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=50, strides=[1, 1], padding="VALID"),
            MaxPool([2, 2], [2, 2], padding="VALID"),
            ReLu(),
            BatchNormalization(momentum=0.9),
            FullyConnected(500, flatten=True),
            ReLu(),
            BatchNormalization(momentum=0.9),
            FullyConnected(output_size)]

def liao_cifar(output_size):
    return [ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=32, strides=[1, 1], padding="SAME"),
            MaxPool([3, 3], [2, 2], padding="SAME"),
            ReLu(),
            ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=64, strides=[1, 1], padding="SAME"),
            AveragePool([3, 3], [2, 2], padding="SAME"),
            ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=64, strides=[1, 1], padding="SAME"),
            AveragePool([3, 3], [2, 2], padding="SAME"),
            ReLu(),
            FullyConnected(128, flatten=True),
            ReLu(),
            FullyConnected(output_size)]

def liao_cifar_bn(output_size):
    return [ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=32, strides=[1, 1], padding="SAME"),
            MaxPool([3, 3], [2, 2], padding="SAME"),
            ReLu(),
            BatchNormalization(momentum=0.9),
            ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=64, strides=[1, 1], padding="SAME"),
            AveragePool([3, 3], [2, 2], padding="SAME"),
            ReLu(),
            BatchNormalization(momentum=0.9),
            ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=64, strides=[1, 1], padding="SAME"),
            AveragePool([3, 3], [2, 2], padding="SAME"),
            ReLu(),
            BatchNormalization(momentum=0.9),
            FullyConnected(128, flatten=True),
            ReLu(),
            BatchNormalization(momentum=0.9),
            FullyConnected(output_size)]

def experiment_mnist_fc(output_size):
    return [FullyConnected(100, flatten=True),
            ReLu(),
            FullyConnected(30),
            ReLu(),
            FullyConnected(output_size)]

def experiment_mnist_conv(output_size):
    return [ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=20, strides=[1, 1], padding="SAME"),
            MaxPool([2, 2], [2, 2]),
            ReLu(),
            ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=50, strides=[1, 1], padding="SAME"),
            MaxPool([2, 2], [2, 2]),
            ReLu(),
            FullyConnected(500, flatten=True),
            ReLu(),
            FullyConnected(output_size)]

def vgg_16(output_size):
    def convolutional_layer_block(filter_dim, number_of_filters, strides=[1, 1], with_pooling=False):
        if with_pooling == False:
            return [ConvolutionalLayer(filter_dim, strides=strides, num_of_filters=number_of_filters, padding="SAME"), ReLu()]
        else:
            return [ConvolutionalLayer(filter_dim, strides=strides, num_of_filters=number_of_filters, padding="SAME"),
                    ReLu(), MaxPool([2, 2], [2, 2], padding="SAME")]

    sequence = []
    sequence.extend(convolutional_layer_block((3, 3), number_of_filters=64))
    sequence.extend(convolutional_layer_block((3, 3), number_of_filters=64, with_pooling=True))

    sequence.extend(convolutional_layer_block((3, 3), number_of_filters=128))
    sequence.extend(convolutional_layer_block((3, 3), number_of_filters=128, with_pooling=True))

    sequence.extend(convolutional_layer_block((3, 3), number_of_filters=256))
    sequence.extend(convolutional_layer_block((3, 3), number_of_filters=256))
    sequence.extend(convolutional_layer_block((3, 3), number_of_filters=256, with_pooling=True))

    sequence.extend(convolutional_layer_block((3, 3), number_of_filters=512))
    sequence.extend(convolutional_layer_block((3, 3), number_of_filters=512))
    sequence.extend(convolutional_layer_block((3, 3), number_of_filters=512, with_pooling=True))

    sequence.extend(convolutional_layer_block((3, 3), number_of_filters=512))
    sequence.extend(convolutional_layer_block((3, 3), number_of_filters=512))
    sequence.extend(convolutional_layer_block((3, 3), number_of_filters=512, with_pooling=True))

    sequence.extend([FullyConnected(4096, flatten=True), ReLu()]),
    sequence.extend([FullyConnected(4096), ReLu()]),
    sequence.extend([FullyConnected(output_size)])
    return sequence
