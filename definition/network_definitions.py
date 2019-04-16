from numpy.random import randint

default_network = {
    "type": "BP",
    "dataset_name": "mnist",
    "sequence": "fc1",
    "cost_function": "mean_squared_error",
    "learning_rate": 0.1,
    "gather_stats": False,
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

main_network_1 = dict(default_network)
main_network_1.update({
    "cost_function": "softmax_cross_entropy",
    "dataset_name": "cifar10",
    "batch_size": 256,
    "minimum_accuracy": [(10, 15)]
})

main_network_2 = dict(main_network_1)
main_network_2.update({
    "batch_size": 128
})

first_dfa_1 = dict(main_network_1)
first_dfa_1.update({
    "learning_rate": 0.003,
    "epochs": 30,
    "type": "DFA",
    "save_model": True,
    "save_model_path": "./saved/model.ckpt"
})

then_bp_1 = dict(main_network_1)
then_bp_1.update({
    "learning_rate": 0.001,
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

main_network_3 = dict(main_network_2)
main_network_3.update({
    "seed": 53220922,
    "sequence": "vgg_16",
})

first_bp_2 = dict(main_network_3)
first_bp_2.update({
    "learning_rate": 0.001,
    "save_model": True,
    "save_model_path": "./saved_2/model.ckpt",
    "epochs": 40
})

then_bp_2 = dict(main_network_3)
then_bp_2.update({
    "learning_rate": 0.0001,
    "restore_model": True,
    "restore_model_path": "./saved_2/model.ckpt",
    "epochs": 40
})

just_bp_2_1 = dict(main_network_3)
just_bp_2_1.update({
    "learning_rate": 0.001,
    "epochs": 80
})

just_bp_2_2 = dict(main_network_3)
just_bp_2_2.update({
    "learning_rate": 0.0001,
    "epochs": 80
})

first_dfa_3 = dict(first_dfa_1)
first_dfa_3.update({
    "epochs": 5,
    "batch_size": 128
})

then_bp_3 = dict(then_bp_1)
then_bp_3.update({
    "epochs": 35,
    "batch_size": 128
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
