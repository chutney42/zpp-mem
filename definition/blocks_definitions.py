from layer.activation.activation_layer import Sigmoid
from layer.block import Block
from layer.util_layer.batch_normalization import BatchNormalization
from layer.util_layer.max_pool import MaxPool
from layer.weight_layer.fully_connected import FullyConnected, FullyConnectedManhattan
from layer.weight_layer.convolutional_layers import ConvolutionalLayer, ConvolutionalLayerManhattan


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
