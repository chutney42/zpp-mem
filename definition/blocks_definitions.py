from definition.resnet import *
from layer import *


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
        [Conv((5, 5), number_of_filters=10), BatchNormalization(), Sigmoid()]),
        Block([Conv((5, 5), number_of_filters=10), BatchNormalization(), Sigmoid()]),
        Block([FullyConnected(30, flatten=True), BatchNormalization(), Sigmoid()]),
        Block([FullyConnected(output_size), Sigmoid()])]

def conv1prim(output_size):
    return [Block(
        [DepthWiseSeperableConv((5, 5), number_of_filters=10), BatchNormalization(), Sigmoid()]),
        Block([DepthWiseSeperableConv((5, 5), number_of_filters=10), BatchNormalization(), Sigmoid()]),
        Block([FullyConnected(30, flatten=True), BatchNormalization(), Sigmoid()]),
        Block([FullyConnected(output_size), Sigmoid()])]


def conv2(output_size):
    return [Block(
        [Conv((5, 5), number_of_filters=10), BatchNormalization(), MaxPool([4, 4], [2, 2]), Sigmoid()]),
        Block([Conv((5, 5), number_of_filters=10), BatchNormalization(), MaxPool([4, 4], [2, 2]),
               Sigmoid()]),
        Block([FullyConnected(output_size, flatten=True), Sigmoid()])]


def conv3(output_size):
    return [Block(
        [Conv((5, 5), number_of_filters=15), BatchNormalization(), MaxPool([4, 4], [2, 2]), Sigmoid()]),
        Block([Conv((5, 5), number_of_filters=15), BatchNormalization(), MaxPool([4, 4], [2, 2]),
               Sigmoid()]),
        Block([Conv((5, 5), number_of_filters=15), BatchNormalization(), MaxPool([4, 4], [2, 2]),
               Sigmoid()]),
        Block([FullyConnected(output_size, flatten=True), Sigmoid()])]


def long_fc(output_size):
    blocks = [Block([FullyConnected(500, flatten=(i == 0)), BatchNormalization(), Sigmoid()]) for i in range(30)]
    blocks.append(Block([FullyConnected(output_size), Sigmoid()]))
    return blocks


def long_conv(output_size):
    blocks = [Block([Conv((5, 5), number_of_filters=5), BatchNormalization(), Sigmoid()]) for i in
              range(30)]
    blocks.append(Block([FullyConnected(output_size, flatten=True), Sigmoid()]))
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


def vgg_16(output_size):
    def convolutional_layer_block(filter_dim, number_of_filters, stride=[1, 1], with_pooling=False):
        if with_pooling == False:
            return Block([Conv(filter_dim, stride=stride, number_of_filters=number_of_filters),
                          BatchNormalization(), ReLu()])
        else:
            return Block([Conv(filter_dim, stride=stride, number_of_filters=number_of_filters),
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
            return Block([Conv(filter_dim, stride=stride, number_of_filters=number_of_filters),
                          ReLu()])
        else:
            return Block([Conv(filter_dim, stride=stride, number_of_filters=number_of_filters),
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

# def mobile_net(output_size):
#