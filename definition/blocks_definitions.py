from layer.activation.activation_layer import Sigmoid, ReLu
from layer.block import Block
from layer.util_layer.batch_normalization import BatchNormalization
from layer.util_layer.max_pool import MaxPool
from layer.weight_layer.fully_connected import FullyConnected, FullyConnectedManhattan
from layer.weight_layer.convolutional_layers import ConvolutionalLayer, ConvolutionalLayerManhattan
from layer.weight_layer.fully_connected import FullyConnected
from layer.util_layer.softmax import Softmax


def fc1(output_size):
    return [Block([FullyConnected(50, flatten=True), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(30), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(output_size), Sigmoid()])]


def fc2(output_size):
    return [Block([FullyConnected(100, flatten=True), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(200), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(50), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(output_size), Sigmoid()])]


def fc3(output_size):
    return [Block([FullyConnected(200, flatten=True), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(1000), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(100), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(output_size), Sigmoid()])]


def fc1_relu(output_size):
    return [Block([FullyConnected(50, flatten=True), BatchNormalization(), ReLu()]),
            Block([FullyConnected(30), BatchNormalization(), ReLu()]),
            Block([FullyConnected(output_size), Sigmoid()])]


def fc2_relu(output_size):
    return [Block([FullyConnected(100, flatten=True), BatchNormalization(), ReLu()]),
            Block([FullyConnected(200), BatchNormalization(), ReLu()]),
            Block([FullyConnected(50), BatchNormalization(), ReLu()]),
            Block([FullyConnected(output_size), Sigmoid()])]


def fc3_relu(output_size):
    return [Block([FullyConnected(200, flatten=True), BatchNormalization(), ReLu()]),
            Block([FullyConnected(1000), BatchNormalization(), ReLu()]),
            Block([FullyConnected(100), BatchNormalization(), ReLu()]),
            Block([FullyConnected(output_size), Sigmoid()])]


def conv1(output_size):
    return [Block(
        [ConvolutionalLayer((5, 5), number_of_filters=10), BatchNormalization(), Sigmoid()]),
        Block([ConvolutionalLayer((5, 5), number_of_filters=10), BatchNormalization(), Sigmoid()]),
        Block([FullyConnected(30, flatten=True), BatchNormalization(), Sigmoid()]),
        Block([FullyConnected(output_size), Sigmoid()])]


def conv2(output_size):
    return [Block(
        [ConvolutionalLayer((5, 5), number_of_filters=10), BatchNormalization(), MaxPool([4, 4], [2, 2]), Sigmoid()]),
        Block([ConvolutionalLayer((5, 5), number_of_filters=10), BatchNormalization(), MaxPool([4, 4], [2, 2]),
               Sigmoid()]),
        Block([FullyConnected(output_size, flatten=True), Sigmoid()])]


def conv3(output_size):
    return [Block(
        [ConvolutionalLayer((5, 5), number_of_filters=15), BatchNormalization(), MaxPool([4, 4], [2, 2]), Sigmoid()]),
        Block([ConvolutionalLayer((5, 5), number_of_filters=15), BatchNormalization(), MaxPool([4, 4], [2, 2]),
               Sigmoid()]),
        Block([ConvolutionalLayer((5, 5), number_of_filters=15), BatchNormalization(), MaxPool([4, 4], [2, 2]),
               Sigmoid()]),
        Block([FullyConnected(output_size, flatten=True), Sigmoid()])]


def long_fc(output_size):
    blocks = [Block([FullyConnected(500, flatten=(i == 0)), BatchNormalization(), Sigmoid()]) for i in range(30)]
    blocks.append(Block([FullyConnected(output_size), Sigmoid()]))
    return blocks


def long_conv(output_size):
    blocks = [Block([ConvolutionalLayer((5, 5), number_of_filters=5), BatchNormalization(), Sigmoid()]) for i in
              range(30)]
    blocks.append(Block([FullyConnected(output_size, flatten=True), Sigmoid()]))
    return blocks
