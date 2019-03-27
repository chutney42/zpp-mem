default_network = {
    "type": "BP",
    "dataset_name": "mnist",
    "sequence": "resnet_18",
    "cost_function": "mean_squared_error",
    "learning_rate": 0.1,
    "gather_stats": False,
    "restore_model": False,
    "save_model": False,
    "restore_model_path": None,
    "save_model_path": None,

    "batch_size": 10,
    "epochs": 4,
    "eval_period": 1000,
    "stat_period": 100,
    "memory_only": False,
    "seed": None,
    "device": "/cpu:0"
}


