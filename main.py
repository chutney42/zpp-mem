import argparse

from layer.block import Block
from layer.util_layer.batch_normalization import BatchNormalization
from layer.weight_layer.convolutional_layers import ConvolutionalLayer
from layer.weight_layer.fully_connected import FullyConnected
from layer.activation.activation_layer import *
from util.options import parse
from definition import dataset_definitions
from inspect import getmembers, isfunction
import time

datasets_dict = dict(getmembers(dataset_definitions, isfunction))


def load_dataset(options):
    dataset = options['dataset']
    if dataset is None:
        dataset_name = 'mnist'
    else:
        dataset_name = options['dataset']['name']
    return datasets_dict[dataset_name]()


def define_network(output_types, output_shapes, options):
    model = options['type']
    if model == 'BP':
        from neural_network.backpropagation import Backpropagation as Network
    elif model == 'DFA':
        from neural_network.direct_feedback_alignment import DirectFeedbackAlignment as Network
    elif model == 'FA':
        from neural_network.feedback_alignment import FeedbackAlignment as Network
    else:
        raise NotImplementedError(f"Model {model} is not recognized.")

    return Network(output_types, output_shapes,
                   [Block([ConvolutionalLayer((3,3), number_of_filters=10), BatchNormalization(), Sigmoid()]),
                    Block([FullyConnected(30,flatten=True), BatchNormalization(), Sigmoid()]),
                    Block([FullyConnected(output_shapes[1][0].value), Sigmoid()])],
                   learning_rate=options['training_parameters']['learning_rate'],
                   scope=options['type'],
                   gather_stats=options['training_parameters']['gather_stats'],
                   restore_model=options['model_handling']['restore'],
                   save_model=options['model_handling']['save'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=False, help='Path to option JSON file.')

    if parser.parse_args().opt is None:
         opt_path = "./options/backpropagation.json"
        # opt_path = "./options/direct_feedback_alignment.json"
        # opt_path = "./options/feedback_alignment.json"
    else:
        opt_path = parser.parse_args().opt

    options = parse(opt_path)

    batch_size = options['dataset']['batch_size']
    epochs = options['dataset']['epochs']
    eval_period = options['periods']['eval_period']
    stat_period = options['periods']['stat_period']

    g = tf.Graph()
    dev = '/cpu:0'
    with tf.device(dev) as de:
        with g.as_default():
            tf.set_random_seed(1)
            training, test = load_dataset(options)
            NN = define_network(training.output_types, training.output_shapes, options)
            start_learning_time = time.time()
            if options['is_train']:
                NN.train(training, test, batch_size=batch_size, epochs=epochs, eval_period=eval_period,
                         stat_period=stat_period)
            else:
                NN.test(test, batch_size)

    print(f"Learning process took {time.time() - start_learning_time} seconds")
