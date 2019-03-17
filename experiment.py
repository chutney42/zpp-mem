import argparse
import time

from inspect import getmembers, isfunction
from exp import blocks_definitions
from exp import network_definitions
from util.loader import datasets

blocks_dict = dict(getmembers(blocks_definitions, isfunction))
networks_dict = {
    network_name: network_definition for network_name, network_definition in getmembers(network_definitions)
    if isinstance(network_definition, dict) and network_name != "__builtins__"
}
networks_list = [
    network_definition for network_name, network_definition in getmembers(network_definitions)
    if isinstance(network_definition, dict) and network_name != "__builtins__"
]


def define_network(network, output_types, output_shapes):
    model = network['type']
    if model == 'BP':
        from neural_network.backpropagation import Backpropagation as Network
    elif model == 'DFA':
        from neural_network.direct_feedback_alignment import DirectFeedbackAlignment as Network
    elif model == 'FA':
        from neural_network.feedback_alignment import FeedbackAlignment as Network
    else:
        raise NotImplementedError(f"Model {model} is not recognized.")

    sequence = blocks_dict[network['sequence']](output_shapes[1][0].value)

    return Network(output_types,
                   output_shapes,
                   sequence,
                   learning_rate=network['learning_rate'],
                   scope=model,
                   gather_stats=network['gather_stats'],
                   # restore_model_path=network['restore_model_path'],
                   # save_model_path=network['save_model_path'],
                   restore_model=network['restore_model'],
                   save_model=network['save_model'])


def learn_network(neural_network, training, test, network):
    start_learning_time = time.time()
    neural_network.train(training_set=training,
                         validation_set=test,
                         batch_size=network['batch_size'],
                         epochs=network['epochs'],
                         eval_period=network['eval_period'],
                         stat_period=network['stat_period'],
                         memory_only=network['memory_only'])
    print(f"learning process took {time.time() - start_learning_time} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', type=int, required=False, help='number of network')
    parser.add_argument('-name', type=str, required=False, help='network name')
    id = parser.parse_args().id
    name = parser.parse_args().name
    if id is not None:
        network = networks_list[id]
    elif name is not None:
        network = networks_dict[name]
    else:
        raise Exception("you must choose a network to run")
    training, test = datasets[network['dataset_name']]()
    neural_network = define_network(network, training.output_types, training.output_shapes)
    learn_network(neural_network, training, test, network)
