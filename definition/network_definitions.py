from numpy.random import randint

default_network = {
    "type": "DFA",
    "dataset_name": "mnist",
    "sequence": "conv0",
    "cost_function": "softmax_cross_entropy",
    "learning_rate": 0.001,
    "gather_stats": False,
    "restore_model": False,
    "save_model": False,
    "restore_model_path": None,
    "save_model_path": None,

    "minimum_accuracy": [(1, 99)],
    "batch_size": 10,
    "epochs": 4,
    "eval_period": 1000,
    "stat_period": 100,
    "memory_only": False,
    "seed": randint(1, 100000000),
}

vgg_16 = dict(default_network)
vgg_16.update({
    "minimum_accuracy": [(10, 12), (50, 20)],
    "type": "BP",
    "sequence": "vgg_16",
    "epochs": 100,
    "cost_function": "softmax_cross_entropy",
    "dataset_name": "cifar10"

})

vgg_16_DFA = dict(vgg_16)
vgg_16_DFA.update({
    "type": "DFA",
    "minimum_accuracy": [(20, 20), (50, 40)],

})


resnet = dict(default_network)
resnet.update({
    "type": "DFA",
    "dataset_name": "cifar10",
    "sequence": "resnet_18",
    "minimum_accuracy": [(10, 20)],
    "epochs": 50,
    "batch_size": 64,
    "cost_function": "softmax_cross_entropy",
})