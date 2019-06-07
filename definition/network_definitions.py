from numpy.random import randint

default_network = {
    "type": "DFAMEM",
    "dataset_name": "mnist",
    "sequence": "long_fc2",

    "cost_function": "softmax_cross_entropy",
    "learning_rate": 0.01,
#    "momentum": 0.9,
    "minimize_manually": True,
    
    "gather_stats": False,
    "save_graph": False,
    "memory_only": True,

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

liao_network = {
    "type": "BP",
    "dataset_name": "mnist",
    "sequence": "liao_mnist",
    "cost_function": "softmax_cross_entropy",
    "learning_rate": 0.005,
#    "momentum": 0.9,
    "minimize_manually": True,

    "gather_stats": False,
    "save_graph": False,
    "memory_only": False,

    "restore_model": False,
    "save_model": False,
    "restore_model_path": None,
    "save_model_path": None,

    "minimum_accuracy": [(1, 1)],
    "batch_size": 100,
    "epochs": 50,
    "eval_period": 1000,
    "stat_period": 100,
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
