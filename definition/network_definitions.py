from numpy.random import randint


default_network = {
    "type": "DFAMEM",
    "dataset_name": "mnist",
    "sequence": "fcReLu",
    "cost_function": "softmax_cross_entropy",
    "learning_rate": 0.01,
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
    "batch_size": 10,
    "epochs": 30,
    "eval_period": 1000,
    "stat_period": 100,
    "seed": randint(1, 100000000),
}

fcSigmoid = dict(default_network)
fcSigmoid.update({
    "sequence": "fcSigmoid",
    "cost_function": "mean_squared_error",
    "learning_reate": 0.05,
    "batch_size": 10,
    "epochs":30,
    "minimize_manually": True,
})

fcReLu = dict(default_network)
fcReLu.update({
    "sequence": "fcReLu",
    "cost_function": "softmax_cross_entropy",
    "learning_reate": 0.01,
    "batch_size": 10,
    "epochs":30,
    "minimize_manually": True,
})


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
    "epochs": 150,
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

exp_mnist_fc = dict(default_network)
exp_mnist_fc.update({
    "sequence": "experiment_mnist_fc",
    "batch_size": 100,
    "epochs": 100
})

exp_mnist_conv = dict(default_network)
exp_mnist_conv.update({
    "sequence": "experiment_mnist_conv",
    "learning_rate": 0.005,
    "batch_size": 100,
    "epochs": 150
})

exp_cifar_vgg_bp = dict(default_network)
exp_cifar_vgg_bp.update({
    "type": "BP",
    "dataset_name": "cifar10",
    "sequence": "vgg_16",
    "cost_function": "softmax_cross_entropy",
    "learning_rate": 0.001,
    "memory_only": False,
    "minimize_manually": False,
    "minimum_accuracy": [(5, 20)],
    "batch_size": 100,
    "epochs": 80,
})

exp_cifar_vgg_fa = dict(exp_cifar_vgg_bp)
exp_cifar_vgg_fa.update({
    "type": "FA",
    "learning_rate": 0.000005
})

exp_cifar_vgg_dfa = dict(exp_cifar_vgg_fa)
exp_cifar_vgg_dfa.update({
    "type": "DFA"
})

exp_cifar_vgg_memdfa = dict(exp_cifar_vgg_fa)
exp_cifar_vgg_memdfa.update({
    "type": "DFAMEM"
})

exp_cifar_mosko = dict(default_network)
exp_cifar_mosko.update({
    "dataset_name": "cifar10",
    "sequence": "experiment_cifar_mosko",
    "learning_rate": 0.01,
    "batch_size": 100,
    "epochs": 5
})

exp_cifar_liao_bp = dict(default_network)
exp_cifar_liao_bp.update({
    "dataset_name": "cifar10",
    "sequence": "liao_cifar",
    "minimum_accuracy": [(3, 11), (10, 20)],
    "learning_rate": 0.001,
    "batch_size": 100,
    "epochs": 100
})


exp_cifar_liao_fa = dict(exp_cifar_liao_bp)
exp_cifar_liao_fa.update({
    "learning_rate": 0.00001,
})

exp_cifar_liao_dfa = dict(exp_cifar_liao_bp)
exp_cifar_liao_dfa.update({
    "learning_rate": 0.00001,
})

exp_cifar_liao_memdfa = dict(exp_cifar_liao_bp)
exp_cifar_liao_memdfa.update({
    "learning_rate": 0.00001,
})

exp_cifar_liao_bn_bp = dict(exp_cifar_liao_bp)
exp_cifar_liao_bn_bp.update({
    "sequence": "liao_cifar_bn"
})

exp_cifar_liao_bn_fa = dict(exp_cifar_liao_bn_bp)
exp_cifar_liao_bn_fa.update({
    "learning_rate": 0.00001,
})

exp_cifar_liao_bn_dfa = dict(exp_cifar_liao_bn_bp)
exp_cifar_liao_bn_dfa.update({
    "learning_rate": 0.00001,
})

exp_cifar_liao_bn_memdfa = dict(exp_cifar_liao_bn_bp)
exp_cifar_liao_bn_memdfa.update({
    "learning_rate": 0.00001,
})