from numpy.random import randint

default_network = {
    "type": "BP",
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

mosco = dict(default_network)
mosco.update({
    "dataset_name": "cifar10",
    "sequence": "moskovitz_cifar_2",
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 10,
    "minimum_accuracy": [(3, 11)],
})

mosco_1 = dict(mosco)
mosco_1.update({
    "learning_rate": 0.1,
    "epochs": 20,
    "save_model": True,
    "save_model_path": "./saved_1/model.ckpt"
})

mosco_2 = dict(mosco)
mosco_2.update({
    "learning_rate": 0.06,
    "epochs": 10,
    "restore_model": True,
    "restore_model_path": "./saved_1/model.ckpt",
    "save_model": True,
    "save_model_path": "./saved_2/model.ckpt"
})


mosco_3 = dict(mosco)
mosco_3.update({
    "learning_rate": 0.03,
    "epochs": 10,
    "restore_model": True,
    "restore_model_path": "./saved_2/model.ckpt",
    "save_model": True,
    "save_model_path": "./saved_3/model.ckpt"
})

liao_network = {
    "type": "DFA",
    "dataset_name": "mnist",
    "sequence": "liao_mnist_bn",
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

moskovitz_2 = dict(default_network)
moskovitz_2.update({
    "sequence": "moskovitz_cifar_2",
    "dataset_name": "cifar10",
    "batch_size": 128
})

first_dfa_1 = dict(moskovitz_2)
first_dfa_1.update({
    "learning_rate": 0.05,
    "epochs": 30,
    "type": "DFA",
    "save_model": True,
    "save_model_path": "./saved/model.ckpt"
})

then_bp_1 = dict(moskovitz_2)
then_bp_1.update({
    "learning_rate": 0.05,
    "epochs": 10,
    "restore_model": True,
    "restore_model_path": "./saved/model.ckpt"
})

just_dfa_1 = dict(first_dfa_1)
just_dfa_1.update({
    "epochs": 40,
    "save_model": False
})

just_bp_1 = dict(then_bp_1)
just_bp_1.update({
    "epochs": 40,
    "restore_model": False
})
