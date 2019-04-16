from definition.resnet import *
from layer import *


def fc0(output_size):
    return [FullyConnected(50, flatten=True), Sigmoid(),
            FullyConnected(30), Sigmoid(),
            FullyConnected(output_size), Sigmoid()]


def fc1(output_size):
    return [FullyConnected(50, flatten=True), BatchNormalization(), Sigmoid(),
            FullyConnected(30), BatchNormalization(), Sigmoid(),
            FullyConnected(output_size), Sigmoid()]


def fc2(output_size):
    return [FullyConnected(100, flatten=True), BatchNormalization(), Sigmoid(),
            FullyConnected(200), BatchNormalization(), Sigmoid(),
            FullyConnected(50), BatchNormalization(), Sigmoid(),
            FullyConnected(output_size), Sigmoid()]


def fc3(output_size):
    return [FullyConnected(200, flatten=True), BatchNormalization(), Sigmoid(),
            FullyConnected(1000), BatchNormalization(), Sigmoid(),
            FullyConnected(100), BatchNormalization(), Sigmoid(),
            FullyConnected(output_size), Sigmoid()]


def fc1_relu(output_size):
    return [FullyConnected(50, flatten=True), BatchNormalization(), ReLu(),
            FullyConnected(30), BatchNormalization(), ReLu(),
            FullyConnected(output_size), Sigmoid()]


def fc2_relu(output_size):
    return [FullyConnected(100, flatten=True), BatchNormalization(), ReLu(),
            FullyConnected(200), BatchNormalization(), ReLu(),
            FullyConnected(50), BatchNormalization(), ReLu(),
            FullyConnected(output_size), Sigmoid()]


def fc3_relu(output_size):
    return [FullyConnected(200, flatten=True), BatchNormalization(), ReLu(),
            FullyConnected(1000), BatchNormalization(), ReLu(),
            FullyConnected(100), BatchNormalization(), ReLu(),
            FullyConnected(output_size), Sigmoid()]


def conv0(output_size):
    return [ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=10, strides=[1, 1, 1, 1], padding="SAME"), Sigmoid(),
            ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=10, strides=[1, 1, 1, 1], padding="SAME"), Sigmoid(),
            FullyConnected(30, flatten=True), Sigmoid(),
            FullyConnected(output_size)]

def conv1(output_size):
    return [ConvolutionalLayer((5, 5), 10, [1, 1, 1, 1]), BatchNormalization(), Sigmoid(),
            ConvolutionalLayer((5, 5), num_of_filters=10), BatchNormalization(), Sigmoid(),
            FullyConnected(30, flatten=True), BatchNormalization(), Sigmoid(),
            FullyConnected(output_size), Sigmoid()]


def conv2(output_size):
    return [ConvolutionalLayer((5, 5), num_of_filters=10), BatchNormalization(), MaxPool([4, 4], [2, 2]), Sigmoid(),
            ConvolutionalLayer((5, 5), num_of_filters=10), BatchNormalization(), MaxPool([4, 4], [2, 2]), Sigmoid(),
            FullyConnected(output_size, flatten=True), Sigmoid()]


def conv3(output_size):
    return [ConvolutionalLayer((5, 5), num_of_filters=15), BatchNormalization(), MaxPool([4, 4], [2, 2]), Sigmoid(),
            ConvolutionalLayer((5, 5), num_of_filters=15), BatchNormalization(), MaxPool([4, 4], [2, 2]), Sigmoid(),
            ConvolutionalLayer((5, 5), num_of_filters=15), BatchNormalization(), MaxPool([4, 4], [2, 2]), Sigmoid(),
            FullyConnected(output_size, flatten=True), Sigmoid()]


def long_fc(output_size):
    sequence = []
    for i in range(30):
        sequence += [FullyConnected(500, flatten=(i == 0)), BatchNormalization(), Sigmoid()]
    sequence += [FullyConnected(output_size), Sigmoid()]
    return sequence


def long_conv(output_size):
    sequence = []
    for i in range(30):
        sequence += [ConvolutionalLayer((5, 5), num_of_filters=5), BatchNormalization(), Sigmoid()]
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

''' TODO
def vgg_16(output_size):
    def convolutional_layer_block(filter_dim, number_of_filters, stride=[1, 1], with_pooling=False):
        if with_pooling == False:
            return Block([ConvolutionalLayer(filter_dim, stride=stride, number_of_filters=number_of_filters),
                          BatchNormalization(), ReLu()])
        else:
            return Block([ConvolutionalLayer(filter_dim, stride=stride, number_of_filters=number_of_filters),
                          BatchNormalization(), ReLu(), MaxPool([2, 2], [2, 2], padding="SAME")])

    return [
        convolutional_layer_block((3, 3), number_of_filters=64),
        convolutional_layer_block((3, 3), number_of_filters=64, with_pooling=True),

        convolutional_layer_block((3, 3), number_of_filters=128),
        convolutional_layer_block((3, 3), number_of_filters=128, with_pooling=True),
        convolutional_layer_block((3, 3), number_of_filters=256),
        convolutional_layer_block((3, 3), number_of_filters=256),
        convolutional_layer_block((3, 3), number_of_filters=256, with_pooling=True),

        convolutional_layer_block((3, 3), number_of_filters=512),
        convolutional_layer_block((3, 3), number_of_filters=512),
        convolutional_layer_block((3, 3), number_of_filters=512, with_pooling=True),

        convolutional_layer_block((3, 3), number_of_filters=512),
        convolutional_layer_block((3, 3), number_of_filters=512),
        convolutional_layer_block((3, 3), number_of_filters=512, with_pooling=True),

        Block([FullyConnected(4096, flatten=True), BatchNormalization(), ReLu()]),
        Block([FullyConnected(4096), BatchNormalization(), ReLu()]),
        Block([FullyConnected(output_size), Softmax()])
    ]


def vgg_16_without_BN(output_size):
    def convolutional_layer_block(filter_dim, number_of_filters, stride=[1, 1], with_pooling=False):
        if with_pooling == False:
            return Block([ConvolutionalLayer(filter_dim, stride=stride, number_of_filters=number_of_filters),
                          ReLu()])
        else:
            return Block([ConvolutionalLayer(filter_dim, stride=stride, number_of_filters=number_of_filters),
                          ReLu(), MaxPool([2, 2], [2, 2], padding="SAME")])

    return [
        convolutional_layer_block((3, 3), number_of_filters=64),
        convolutional_layer_block((3, 3), number_of_filters=64, with_pooling=True),

        convolutional_layer_block((3, 3), number_of_filters=128),
        convolutional_layer_block((3, 3), number_of_filters=128, with_pooling=True),
        convolutional_layer_block((3, 3), number_of_filters=256),
        convolutional_layer_block((3, 3), number_of_filters=256),
        convolutional_layer_block((3, 3), number_of_filters=256, with_pooling=True),

        convolutional_layer_block((3, 3), number_of_filters=512),
        convolutional_layer_block((3, 3), number_of_filters=512),
        convolutional_layer_block((3, 3), number_of_filters=512, with_pooling=True),

        convolutional_layer_block((3, 3), number_of_filters=512),
        convolutional_layer_block((3, 3), number_of_filters=512),
        convolutional_layer_block((3, 3), number_of_filters=512, with_pooling=True),

        Block([FullyConnected(4096, flatten=True), BatchNormalization(), ReLu()]),
        Block([FullyConnected(4096), BatchNormalization(), ReLu()]),
        Block([FullyConnected(output_size), Softmax()])
    ]
'''
