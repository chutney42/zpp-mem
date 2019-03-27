import argparse
import time
import tensorflow as tf
from inspect import getmembers, isfunction
from definition import blocks_definitions
from definition import dataset_definitions
from definition import network_definitions

blocks_dict = dict(getmembers(blocks_definitions, isfunction))
datasets_dict = dict(getmembers(dataset_definitions, isfunction))
networks_dict = {
    network_name: network_definition for network_name, network_definition in getmembers(network_definitions)
    if isinstance(network_definition, dict) and network_name != "__builtins__"
}
networks_list = [
    network_definition for network_name, network_definition in getmembers(network_definitions)
    if isinstance(network_definition, dict) and network_name != "__builtins__"
]


def get_id_and_name_from_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', type=int, required=False, help='number of network')
    parser.add_argument('-name', type=str, required=False, help='network name')

    parser.add_argument('-learning_type', type=str, required=False, help='type')
    parser.add_argument('-learning_rate', type=float, required=False, help='learning rate')
    parser.add_argument('-batch_size', type=int, required=False, help='batch size')
    parser.add_argument('-epochs', type=int, required=False, help='epochs')
    parser.add_argument('-cost_method', type=str, required=False, help='cost method')
    parser.add_argument('-dataset', type=str, required=False, help='dataset')

    network_id = parser.parse_args().id
    network_name = parser.parse_args().name
    if network_id is not None and network_name is not None:
        raise Exception("either id or name should be provided, but not both")
    return network_id, network_name, parser


def get_network_definition():
    network_id, network_name, parser = get_id_and_name_from_arguments()

    learning_type = parser.parse_args().learning_type
    learning_rate = parser.parse_args().learning_rate
    batch_size = parser.parse_args().batch_size
    epochs = parser.parse_args().epochs
    cost_method = parser.parse_args().cost_method
    dataset = parser.parse_args().dataset

    if network_id is not None:
        print(f"running network with id={network_id}")
        network_definition = dict(networks_list[network_id])
    elif network_name is not None:
        print(f"running network with name={network_name}")
        network_definition = dict(networks_dict[network_name])
    else:
        raise Exception("you must choose a network to run")

    if learning_type is not None:
        network_definition.update({"type": learning_type})
    if learning_rate is not None:
        network_definition.update({"learning_rate": learning_rate})
    if batch_size is not None:
        network_definition.update({"batch_size": batch_size})
    if epochs is not None:
        network_definition.update({"epochs": epochs})
    if cost_method is not None:
        network_definition.update({"cost_function": cost_method})
    if dataset is not None:
        network_definition.update({"dataset": dataset})

    print(network_definition)
    return network_definition


def create_network(network_definition, output_types, output_shapes):
    model = network_definition['type']
    if model == 'BP':
        from neural_network.backpropagation import Backpropagation as Network
    elif model == 'DFA':
        from neural_network.direct_feedback_alignment import DirectFeedbackAlignment as Network
    elif model == 'FA':
        from neural_network.feedback_alignment import FeedbackAlignment as Network
    else:
        raise NotImplementedError(f"Model {model} is not recognized.")

    sequence = blocks_dict[network_definition['sequence']](output_shapes[1][0].value)

    return Network(output_types,
                   output_shapes,
                   sequence,
                   network_definition['cost_function'],
                   learning_rate=network_definition['learning_rate'],
                   scope=model,
                   gather_stats=network_definition['gather_stats'],
                   # restore_model_path=network['restore_model_path'],
                   # save_model_path=network['save_model_path'],
                   restore_model=network_definition['restore_model'],
                   save_model=network_definition['save_model'])


def train_network(neural_network, training, test, network):
    start_learning_time = time.time()
    neural_network.train(training_set=training,
                         validation_set=test,
                         batch_size=network['batch_size'],
                         epochs=network['epochs'],
                         eval_period=network['eval_period'],
                         stat_period=network['stat_period'],
                         memory_only=network['memory_only'])
    print(f"learning process took {time.time() - start_learning_time} seconds (realtime)")


if __name__ == '__main__':
    network_def = get_network_definition()
    if network_def['seed'] is not None:
        tf.set_random_seed(network_def['seed'])

    training_set, test_set = datasets_dict[network_def['dataset_name']]()
    neural_net = create_network(network_def, training_set.output_types, training_set.output_shapes)
    train_network(neural_net, training_set, test_set, network_def)
