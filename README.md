# ZPP-MEM

## Instalation

To install our project, all you have to do is set up Conda virtual environment.
Conda can be installed from https://docs.conda.io/en/latest/.
Once you have Conda working, you need to create virtual environment and install all required packages.
```bash
$ conda create --name env_name --file requirements.txt
$ conda activate env_name
```
You can deactivate this environment with command
```bash
$ conda deactivate
```
## Defining architecture
In order to define new network's  architecture, in file `./definition/sequence_definitions` you have to define a function  which returns sequence of layers.
```python
def liao_cifar_bn(output_size):
    return [ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=32, strides=[1, 1],
                padding="SAME"), 
            MaxPool(kernel_size=[3, 3], strides=[2, 2], padding="SAME"),
            ReLu(),
            BatchNormalization(momentum=0.9),
            ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=64, strides=[1, 1],
                padding="SAME"),
            AveragePool(kernel_size=[3, 3], strides=[2, 2], padding="SAME"),
            ReLu(),
            BatchNormalization(momentum=0.9),
            ConvolutionalLayer(filter_dim=(5, 5), num_of_filters=64, strides=[1, 1],
                padding="SAME"),
            AveragePool(kernel_size=[3, 3], strides=[2, 2], padding="SAME"),
            ReLu(),
            BatchNormalization(momentum=0.9),
            FullyConnected(128, flatten=True),
            ReLu(),
            BatchNormalization(momentum=0.9),
            FullyConnected(output_size)]
```
## Defining network
In order to define new neural network, in file `./definition/network_definitions` you have to define a dict describing the network.

```python
default_network = {
    "type": "DFA",
    "dataset_name": "mnist",
    "sequence": "conv1",
    "cost_function": "softmax_cross_entropy",
    "learning_rate": 0.01,
    "momentum": 0.9,
    
    "gather_stats": False,
    "save_graph": False,
    "memory_only": False,

    "restore_model": False,
    "save_model": False,
    "restore_model_path": None,
    "save_model_path": None,

    "minimum_accuracy": [(1, 1)],
    "batch_size": 10,
    "epochs": 4,
    "eval_period": 1000,
    "stat_period": 100,
    "seed": randint(1, 100000000),
}
```
You can create new network by extending already existing one.
```python

liao_network = dict(default_network)
liao_network.update({
    "dataset_name": "cifar10",
    "sequence": "liao_cifar_bn",

    "minimum_accuracy": [(10, 30)],
    "batch_size": 100,
    "epochs": 50,
    "eval_period": 1000,
    "stat_period": 100,
    "seed": 1,
})
```
## Running single experiment

Script for running single experiment is called `experiment.py`.
For example, you can run `default_network` with learning algorithm `MEM-DFA` like that:
```bash
$ python experiment.py --name default_network --type DFAMEM
```
You can type `python experiment.py --help` to get list of all possible arguments.

## Running many experiments

There is dedicated script for tuning hiperparameters, called `tuner.py`.
First, you have to prepare this file with hiperparameters you want to test
```python
tuner = cmd_generator(
    [sgen("name", ["default_network", "vgg16"]),
     sgen("type", ["BP", "DFA"]),
     sgen("batch_size", [50, 100, 200],
     sgen("learning_rate", [0.01, 0.001, 0.0001]))],
    command_prefix="python experiment.py",
    output_path=output_path).run_commands()
```
Then, you can simple run this script with `python tuner.py`, and wait for the results.

## Accuracy measuments

Network accuracy when run with `experiment.py` script is printed to stdout.
However, when run with `tuner.py`, all output is redirected to `/hyperparameter/results/<timestamp>`.

## Memory tracing

We implemented possibility to trace memory usage profile durning single iteration of training.

Change flag in configuration
```python
    "memory_only": True
```
Your run session will be interputted after first run and in directory `./plots/` there will be file *.png with plot and *.txt with raw data to analise.

Raw data format:
```csv
1560104632231075 215296 0 forward/DFA_fully_connected_layer_8/fa_fc/IdentityN
1560104632231086 235520 20224 forward/DFA_fully_connected_layer_8/Add
1560104632231104 235520 0 forward/DFA_sigmoid_layer_9/Sigmoid
```
- First column is timestamp in microseconds.
- Second is current memory usage.
- Third is change of usage introduced in given operation.
- Last column is name of operation which invoked memory change. Refer to computation graph created in `./demo` by setting flag:
```python
    "save_graph": True
```
