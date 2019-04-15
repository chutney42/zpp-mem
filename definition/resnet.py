from layer import *


def batch_relu_conv(num_filters, strides=[1, 1, 1, 1]):
    return [BatchNormalization(), ReLu(), ConvolutionalLayer((3, 3), num_of_filters=num_filters, strides=strides)]


def batch_relu_conv_3(num_filters, strides=[1, 1, 1, 1]):
    return [BatchNormalization(), ReLu(), ConvolutionalLayer((1, 1), num_of_filters=num_filters, strides=strides),
            BatchNormalization(), ReLu(), ConvolutionalLayer((3, 3), num_of_filters=num_filters),
            BatchNormalization(), ReLu(), ConvolutionalLayer((1, 1), num_of_filters=num_filters * 4),
            ]


def build_resnet(output_size, size_seq, convolution_generator):
    num_filters = 64

    residual_seq = [ConvolutionalLayer((3, 3), num_of_filters=num_filters)]
    for _ in range(size_seq[0] - 1):
        residual_seq += convolution_generator(num_filters)

    sequence = [ConvolutionalLayer((7, 7), num_of_filters=64, strides=[1, 2, 2, 1]), BatchNormalization(), ReLu(),
                MaxPool([3, 3], strides=[2, 2]), ResidualLayer(residual_seq)]

    for seq in size_seq[1:-1]:
        num_filters *= 2
        residual_seq = convolution_generator(num_filters, [1, 2, 2, 1])
        for _ in range(seq - 1):
            residual_seq += convolution_generator(num_filters)
        sequence.append(ResidualLayer(residual_seq))

    num_filters *= 2
    residual_seq = convolution_generator(num_filters, [1, 2, 2, 1])
    for _ in range(size_seq[-1] - 1):
        residual_seq += convolution_generator(num_filters)
    sequence += [ResidualLayer(residual_seq), BatchNormalization(), ReLu(), AveragePool(None, [1, 1])]
    sequence += [FullyConnected(output_size, flatten=True), Softmax()]
    return sequence
