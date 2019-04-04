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

    "minimum_accuracy": [],
    "batch_size": 10,
    "epochs": 4,
    "eval_period": 1000,
    "stat_period": 100,
    "memory_only": False,
    "seed": None,
    "device": "/cpu:0"
}

vgg_16 = dict(default_network)
vgg_16.update({
    "minimum_accuracy": [(10, 11), (50, 20)],
    "learning_method": "BP",
    "sequence": "vgg_16",
    "epochs": 100,
    "cost_function": "softmax_cross_entropy",
    "dataset_name": "cifar10"

})

vgg_16_DFA = dict(vgg_16)
vgg_16_DFA.update({
    "learning_method": "DFA",
    "minimum_accuracy": [(10, 20), (50, 40)],

})
