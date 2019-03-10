import argparse

from layer import *
from options import parse
from loader import load


def load_data(opt):
    dataset = opt['datasets']
    if dataset is None:
        raise RuntimeError("need dataset")
    return load(dataset['name'])


def define_network(opt):
    if opt['type'] is 'BP':
        from backpropagation import BPNeuralNetwork as Network
    elif opt['type'] is 'DFA':
        from direct_feedback_alignment import DFANeuralNetwork as Network
    elif opt['type'] is 'FA':
        from feedback_alignment import FANeuralNetwork as Network

    return Network(784,
                   [FullyConnected(50),
                    BatchNormalization(),
                    Sigmoid(),
                    FullyConnected(30),
                    BatchNormalization(),
                    Sigmoid(),
                    FullyConnected(10),
                    Sigmoid()],
                   10,
                   learning_rate=opt['training_parameters']['learning_rate'],
                   scope=opt['type'],
                   gather_stats=opt['training_parameters']['gather_stats'],
                   restore_model=opt['model_handling']['restore'],
                   save_model=opt['model_handling']['should_save'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    opt = parse(parser.parse_args().opt)
    print(opt['aalal'])

    training, test = load_data(opt)
    batch_size = opt['datasets']['batch_size']
    epochs = opt['datasets']['epoch']
    eval_period = opt['periods']['eval_period']
    stat_period = opt['periods']['stat_period']

    NN = define_network(opt)
    if opt['is_train']:
        NN.train(training, test, batch_size=batch_size, epochs=epochs, eval_period=eval_period,
                 stat_period=stat_period)
    else:
        NN.test(test, batch_size)
