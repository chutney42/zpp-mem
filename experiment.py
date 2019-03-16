import argparse
from layer.block import Block
from layer.util_layer.batch_normalization import BatchNormalization
from layer.weight_layer.fully_connected import FullyConnected
from layer.activation.activation_layer import *
from neural_network.backpropagation import Backpropagation
from neural_network.feedback_alignment import FeedbackAlignment
from neural_network.direct_feedback_alignment import DirectFeedbackAlignment
from util.loader import load


def create_blocks(id):
    if 0 <= id < 3:
        return [Block([FullyConnected(50), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(30), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(10), Sigmoid()])]
    elif 3 <= id < 9:
        return [Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(500), BatchNormalization(), Sigmoid()]),
                Block([FullyConnected(10), Sigmoid()])]


def create_network(id, training, test):
    if 0 <= id < 3:
        if id == 0:
            return lambda: Backpropagation(training.output_types, training.output_shapes, create_blocks(id), gather_stats=True, scope='BP')\
                .train(training, test, epochs=1, memory_only=True)
        elif id == 1:
            return lambda: FeedbackAlignment(training.output_types, training.output_shapes, create_blocks(id), gather_stats=True, scope='FA')\
                .train(training, test, epochs=1, memory_only=True)
        elif id == 2:
            return lambda: DirectFeedbackAlignment(training.output_types, training.output_shapes, create_blocks(id), gather_stats=True, scope='DFA')\
                .train(training, test, epochs=1, memory_only=True)
    elif 3 <= id < 6:
        if id == 3:
            return lambda: Backpropagation(training.output_types, training.output_shapes, create_blocks(id), gather_stats=True, scope='BP')\
                .train(training, test, epochs=1, memory_only=True)
        elif id == 4:
            return lambda: FeedbackAlignment(training.output_types, training.output_shapes, create_blocks(id), gather_stats=True, scope='FA')\
                .train(training, test, epochs=1, memory_only=True)
        elif id == 5:
            return lambda: DirectFeedbackAlignment(training.output_types, training.output_shapes, create_blocks(id), gather_stats=True, scope='DFA')\
                .train(training, test, epochs=1, memory_only=True)
    elif 6 <= id < 9:
        if id == 6:
            return lambda: Backpropagation(training.output_types, training.output_shapes, create_blocks(id), gather_stats=True, scope='BP')\
                .train(training, test, epochs=1, memory_only=True, batch_size=200)
        elif id == 7:
            return lambda: FeedbackAlignment(training.output_types, training.output_shapes, create_blocks(id), gather_stats=True, scope='FA')\
                .train(training, test, epochs=1, memory_only=True, batch_size=200)
        elif id == 8:
            return lambda: DirectFeedbackAlignment(training.output_types, training.output_shapes, create_blocks(id), gather_stats=True, scope='DFA')\
                .train(training, test, epochs=1, memory_only=True, batch_size=200)
    raise Exception("wrong argument")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', type=int, required=True, help='Number of network')
    id = parser.parse_args().id
    if id == 1:
        exit(-1)
    training, test = load('mnist')
    create_network(id, training, test)()
