from layer.activation.activation_layer import Sigmoid, ReLu
from layer.block import Block
from layer.util_layer.batch_normalization import BatchNormalization
from layer.util_layer.max_pool import MaxPool, AveragePool
from layer.weight_layer.fully_connected import FullyConnected, FullyConnectedManhattan
from layer.weight_layer.convolutional_layers import ConvolutionalLayer, ConvolutionalLayerManhattan
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


def blocks_3x50_10_residual_relu(output_size):
    num_filters = 16
    blocks = [Block([ConvolutionalLayer((7, 7), number_of_filters=16, stride=[2, 2]), MaxPool([3, 3], strides=[2, 2])]),
              Block([ResidualLayer(
                  [Block([BatchNormalization(), ReLu(), ConvolutionalLayer((3, 3), number_of_filters=num_filters)]),
                   Block([BatchNormalization(), ReLu(), ConvolutionalLayer((3, 3), stride=[2, 2], number_of_filters=num_filters)])])])]

    blocks += [Block([ResidualLayer([Block([BatchNormalization(), ReLu(), ConvolutionalLayer((3, 3), stride=[2, 2],
                                                                                             number_of_filters=2 ^ i * num_filters)]),
                                     Block([BatchNormalization(), ReLu(), ConvolutionalLayer((3, 3), stride=[2, 2],
                                                                                             number_of_filters=2 ^ i * num_filters)])])])
               for i in range(1, 3)]
    blocks += [Block([ResidualLayer([Block([BatchNormalization(), ReLu(), ConvolutionalLayer((3, 3), stride=[2, 2],
                                                                                      number_of_filters=2 ^ 4 * num_filters)]),
                              Block([BatchNormalization(), ReLu(), ConvolutionalLayer((3, 3), stride=[2, 2],
                                                                                      number_of_filters=2 ^ 4 * num_filters)])]),
               BatchNormalization(), ReLu(), AveragePool([1, 1], [1, 1])])]
    blocks += [Block([FullyConnected(output_size, flatten=True)]), Sigmoid()]
    return blocks

def blocks_simple_convoluted(output_size):
   return [Block([ConvolutionalLayer((5, 5), number_of_filters=10), BatchNormalization(), MaxPool([4,4],[2,2]), ReLu()]),
           Block([ConvolutionalLayer((5, 5), number_of_filters=10), BatchNormalization(),MaxPool([4,4],[2,2]), ReLu()]),
           Block([FullyConnected(output_size,flatten=True), Sigmoid()])]
