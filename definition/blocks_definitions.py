from definition.resnet import *
from layer.activation.activation_layer import Sigmoid, ReLu
from layer.block import Block
from layer.util_layer.batch_normalization import BatchNormalization
from layer.util_layer.pool import MaxPool
from layer.weight_layer.convolutional_layers import ConvolutionalLayer, ConvolutionalLayerManhattan
from layer.weight_layer.fully_connected import FullyConnected
from layer.weight_layer.fully_connected import FullyConnectedManhattan
from layer.weight_layer.residual_layer import ResidualLayer


def blocks_50_30_10_bn_sigmoid(output_size):
    return [Block([FullyConnected(50, flatten=True), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(30), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(output_size), Sigmoid()])]


def blocks_50_30_10_bn_bm_sigmoid(output_size):
    return [Block([FullyConnectedManhattan(50, flatten=True), BatchNormalization(), Sigmoid()]),
            Block([FullyConnectedManhattan(30), BatchNormalization(), Sigmoid()]),
            Block([FullyConnectedManhattan(output_size), Sigmoid()])]


def blocks_simple_convoluted_with_pool(output_size):
    return [Block(
        [ConvolutionalLayer((3, 3), number_of_filters=5), MaxPool([4, 4], [2, 2]), BatchNormalization(), Sigmoid()]),
        Block([ConvolutionalLayer((3, 3), number_of_filters=5), BatchNormalization(), Sigmoid()]),
        Block([FullyConnected(30, flatten=True), BatchNormalization(), Sigmoid()]),
        Block([FullyConnected(output_size), Sigmoid()])]


def blocks_simple_convoluted(output_size):
    return [Block(
        [ConvolutionalLayer((3, 3), number_of_filters=5), BatchNormalization(), Sigmoid()]),
        Block([ConvolutionalLayer((3, 3), number_of_filters=5), BatchNormalization(), Sigmoid()]),
        Block([FullyConnected(30, flatten=True), BatchNormalization(), Sigmoid()]),
        Block([FullyConnected(output_size), Sigmoid()])]


def blocks_simple_convoluted_bm(output_size):
    return [Block([ConvolutionalLayerManhattan((3, 3), number_of_filters=5), BatchNormalization(), Sigmoid()]),
            Block([ConvolutionalLayerManhattan((3, 3), number_of_filters=5), BatchNormalization(), Sigmoid()]),
            Block([FullyConnectedManhattan(30, flatten=True), BatchNormalization(), Sigmoid()]),
            Block([FullyConnectedManhattan(output_size), Sigmoid()])]


def blocks_30x500_10_bn_sigmoid(output_size):
    blocks = [Block([FullyConnected(500, flatten=(i == 0)), BatchNormalization(), Sigmoid()]) for i in range(30)]
    blocks.append(Block([FullyConnected(output_size), Sigmoid()]))
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


def mini_resnet(output_size):
    num_filters = 16
    blocks = [Block([ConvolutionalLayer((7, 7), number_of_filters=16, stride=[2, 2]), MaxPool([3, 3], strides=[2, 2])]),
              Block([ResidualLayer(
                  [Block([BatchNormalization(), ReLu(), ConvolutionalLayer((3, 3), number_of_filters=num_filters)]),
                   Block([BatchNormalization(), ReLu(),
                          ConvolutionalLayer((3, 3), stride=[2, 2], number_of_filters=num_filters)])]),
                  BatchNormalization(), Sigmoid()])]

    blocks += [Block([FullyConnected(output_size, flatten=True), Sigmoid()])]
    return blocks


def blocks_simple_convoluted(output_size):
    return [Block(
        [ConvolutionalLayer((5, 5), number_of_filters=10), BatchNormalization(), MaxPool([4, 4], [2, 2]), ReLu()]),
        Block([ConvolutionalLayer((5, 5), number_of_filters=10), BatchNormalization(), MaxPool([4, 4], [2, 2]),
               ReLu()]),
        Block([FullyConnected(output_size, flatten=True), Sigmoid()])]
