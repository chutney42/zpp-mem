from layer import *


def batch_relu_conv(num_filters, stride=[1, 1]):
    return [BatchNormalization(), ReLu(), ConvolutionalLayer((3, 3), number_of_filters=num_filters, stride=stride)]


def batch_relu_conv_3(num_filters, stride=[1, 1]):
    return [BatchNormalization(), ReLu(), ConvolutionalLayer((1, 1), number_of_filters=num_filters, stride=stride),
            BatchNormalization(), ReLu(), ConvolutionalLayer((3, 3), number_of_filters=num_filters),
            BatchNormalization(), ReLu(), ConvolutionalLayer((1, 1), number_of_filters=num_filters * 4),
            ]


def build_resnet(output_size, size_seq, convolution_generator):
    num_filters = 64

    residual_seq = [ConvolutionalLayer((3, 3), number_of_filters=num_filters)]
    for _ in range(size_seq[0] - 1):
        residual_seq += convolution_generator(num_filters)

    blocks = [Block([ConvolutionalLayer((7, 7), number_of_filters=64, stride=[2, 2]), BatchNormalization(), ReLu(),
                     MaxPool([3, 3], strides=[2, 2])]),
              Block([ResidualLayer(residual_seq)])]

    for seq in size_seq[1:-1]:
        num_filters *= 2
        residual_seq = convolution_generator(num_filters, [2, 2])
        for _ in range(seq - 1):
            residual_seq += convolution_generator(num_filters)
        blocks += [Block([ResidualLayer(residual_seq)])]

    num_filters *= 2
    residual_seq = convolution_generator(num_filters, [2, 2])
    for _ in range(size_seq[-1] - 1):
        residual_seq += convolution_generator(num_filters)
    blocks += [Block([ResidualLayer(residual_seq), BatchNormalization(), ReLu(), AveragePool(None, [1, 1])])]
    blocks += [Block([FullyConnected(output_size, flatten=True), Softmax()])]
    return blocks
