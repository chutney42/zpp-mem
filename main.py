import argparse

from layer import *
from activation_function import *
from options import parse
from loader import load


def load_dataset(options):
    dataset = options['dataset']
    if dataset is None:
        raise RuntimeError("need dataset")
    return load(dataset['name'])


def define_network(options):
    model = options['type']
    if model == 'BP':
        from backpropagation import Backpropagation as Network
    elif model == 'DFA':
        from direct_feedback_alignment import DirectFeedbackAlignment as Network
    elif model == 'FA':
        from feedback_alignment import FeedbackAlignment as Network
    else:
        raise NotImplementedError(f"Model {model} is not recognized.")

    return Network(784,
                   [Block([FullyConnected(50), BatchNormalization(), Sigmoid()]),
                    Block([FullyConnected(30), BatchNormalization(), Sigmoid()]),
                    Block([FullyConnected(10), Sigmoid()])],
                   10,
                   learning_rate=options['training_parameters']['learning_rate'],
                   scope=options['type'],
                   gather_stats=options['training_parameters']['gather_stats'],
                   restore_model=options['model_handling']['restore'],
                   save_model=options['model_handling']['save'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=False, help='Path to option JSON file.')

    if parser.parse_args().opt is None:
        #opt_path = "./options/backpropagation.json"
        #opt_path = "./options/direct_feedback_alignment.json"
        opt_path = "./options/feedback_alignment.json"
    else:
        opt_path = parser.parse_args().opt

    options = parse(opt_path)

    training, test = load_dataset(options)
    batch_size = options['dataset']['batch_size']
    epochs = options['dataset']['epochs']
    eval_period = options['periods']['eval_period']
    stat_period = options['periods']['stat_period']

    NN = define_network(options)
    if options['is_train']:
        NN.train(training, test, batch_size=batch_size, epochs=epochs, eval_period=eval_period,
                 stat_period=stat_period)
    else:
        NN.test(test, batch_size)
