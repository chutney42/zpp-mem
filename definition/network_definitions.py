from numpy.random import randint

default_network = {
    "type": "BP",
    "dataset_name": "mnist",
    "sequence": "resnet_18",
    "cost_function": "softmax_cross_entropy",
    "learning_rate": 0.1,
    "gather_stats": True,
    "restore_model": False,
    "save_model": False,
    "restore_model_path": None,
    "save_model_path": None,
    "batch_size": 10,
    "epochs": 4,
    "eval_period": 1000,
    "stat_period": 100,
    "memory_only": False,
    "seed": 16219482, #randint(1, 100000000),
    "device": "/cpu:0"
}


