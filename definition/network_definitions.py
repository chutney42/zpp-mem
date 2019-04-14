from numpy.random import randint

default_network = {
    "type": "BP",
    "dataset_name": "mnist",
    "sequence": "fc1",
    "cost_function": "mean_squared_error",
    "learning_rate": 0.1,
    "gather_stats": True,
    "restore_model": False,
    "save_model": False,
    "restore_model_path": None,
    "save_model_path": None,
    "propagator_initializer": "uniform",
    "momentum": 0.9,

    "minimum_accuracy": [(1, 99)],
    "batch_size": 10,
    "epochs": 4,
    "eval_period": 1000,
    "stat_period": 100,
    "memory_only": False,
    "seed": randint(1, 100000000),
    "device": "/cpu:0"
}

main_network = dict(default_network)
main_network.update({
    "dataset_name": "cifar10",
    "batch_size": 128,
    "minimum_accuracy": [],

})

first_dfa = dict(main_network)
first_dfa.update({
    "learning_rate": 0.001,
    "epochs": 7,
    "type": "DFA",
    "save_model": True,
    "save_model_path": "./saved/model.ckpt"
})

then_bp = dict(main_network)
then_bp.update({
    "learning_rate": 0.0005,
    "epochs": 3,
    "restore_model": True,
    "restore_model_path": "./saved/model.ckpt"
})

just_dfa = dict(first_dfa)
just_dfa.update({
    "epochs": 10,
    "save_model": False
})

just_bp = dict(then_bp)
just_bp.update({
    "epochs": 10,
    "restore_model": False
})

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
