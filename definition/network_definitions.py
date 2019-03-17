default_network = {
    # constructor parameters
    "type": "BP",
    "dataset_name": "mnist",
    "sequence": "blocks_1",
    "learning_rate": 0.1,
    "gather_stats": False,
    "restore_model": False,
    "save_model": False,
    # "restore_model_path": "./saved_model/model.ckpt",
    # "save_model_path": "./saved_model/model.ckpt",

    # train parameters
    "batch_size": 20,
    "epochs": 2,
    "eval_period": 1000,
    "stat_period": 100,
    "memory_only": False
}

example_network = dict(default_network)
example_network.update({
    "sequence": "blocks_2",
    "memory_only": True
})
