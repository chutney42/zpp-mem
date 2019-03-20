from layer.activation.activation_layer import Sigmoid
from layer.block import Block
from layer.util_layer.batch_normalization import BatchNormalization
from layer.weight_layer.fully_connected import FullyConnected
from layer.weight_layer.convolutional_layers import ConvolutionalLayer


def blocks_50_30_10_bn_sigmoid(output_size):
    return [Block([FullyConnected(50, flatten=True), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(30), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(output_size), Sigmoid()])]


def blocks_simple_convoluted(output_size):
    return [Block([ConvolutionalLayer((3, 3), number_of_filters=5), BatchNormalization(), Sigmoid()]),
            Block([ConvolutionalLayer((3, 3), number_of_filters=5), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(30,flatten=True), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(output_size), Sigmoid()])]


def blocks_30x500_10_bn_sigmoid(output_size):
    blocks = [Block([FullyConnected(500, flatten=(i == 0)), BatchNormalization(), Sigmoid()]) for i in range(30)]
    blocks.append(Block([FullyConnected(output_size), Sigmoid()]))
    return blocks
