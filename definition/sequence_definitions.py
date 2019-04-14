import tensorflow as tf
from definition.resnet import *
from layer import *


def fc0(output_size):
    return [FullyConnected(50, flatten=True), Sigmoid(),
            FullyConnected(30), Sigmoid(),
            FullyConnected(output_size), Sigmoid()]

'''
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


def conv1(output_size):
    return [ConvolutionalLayer((5, 5), number_of_filters=10), BatchNormalization(), Sigmoid(),
            ConvolutionalLayer((5, 5), number_of_filters=10), BatchNormalization(), Sigmoid(),
            FullyConnected(30, flatten=True), BatchNormalization(), Sigmoid(),
            FullyConnected(output_size), Sigmoid()]


def conv2(output_size):
    return [ConvolutionalLayer((5, 5), number_of_filters=10), BatchNormalization(), MaxPool([4, 4], [2, 2]), Sigmoid(),
            ConvolutionalLayer((5, 5), number_of_filters=10), BatchNormalization(), MaxPool([4, 4], [2, 2]), Sigmoid(),
            FullyConnected(output_size, flatten=True), Sigmoid()]


def conv3(output_size):
    return [ConvolutionalLayer((5, 5), number_of_filters=15), BatchNormalization(), MaxPool([4, 4], [2, 2]), Sigmoid(),
            ConvolutionalLayer((5, 5), number_of_filters=15), BatchNormalization(), MaxPool([4, 4], [2, 2]), Sigmoid(),
            ConvolutionalLayer((5, 5), number_of_filters=15), BatchNormalization(), MaxPool([4, 4], [2, 2]), Sigmoid(),
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
        sequence += [ConvolutionalLayer((5, 5), number_of_filters=5), BatchNormalization(), Sigmoid()]
    sequence += [FullyConnected(output_size, flatten=True), Sigmoid()]
    return blocks


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

'''
