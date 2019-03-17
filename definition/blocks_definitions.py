from layer.activation.activation_layer import Sigmoid
from layer.block import Block
from layer.util_layer.batch_normalization import BatchNormalization
from layer.weight_layer.fully_connected import FullyConnected


def blocks_1(output_size):
    return [Block([FullyConnected(50), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(30), BatchNormalization(), Sigmoid()]),
            Block([FullyConnected(output_size), Sigmoid()])]


def blocks_2(output_size):
        blocks = [Block([FullyConnected(500), BatchNormalization(), Sigmoid()]) for _ in range(30)]
        blocks.append(Block([FullyConnected(output_size), Sigmoid()]))
        return blocks
