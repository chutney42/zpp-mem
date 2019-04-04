import os

from hyperparameter_tuner.run_command_generator import run_command_generator
from hyperparameter_tuner.single_parameter_generator.single_parameter_generator import \
    single_parameter_generator as sgen
import time

# total accuracy: 69.23999786376953% iterations: 16000
# learning process took 63.48976540565491 seconds (realtime)

if __name__ == '__main__':
    output_path = "hyperparameter_tuner/results"
    vgg_16_BP_tuner = run_command_generator([sgen("name", ["vgg_16"]),
                                             sgen("learning_rate", [0.001, 0.005, 0.01, 0.03, 0.07, 0.1, 0.5, 1]),
                                             sgen("batch_size", [20, 25, 30, 35, 50, 75]),
                                             ], command_prefix="python experiment.py",
                                            output_path=output_path).run_commands()

    for command in vgg_16_BP_tuner:
        print(command)
        # os.system("echo $CUDA_VISIBLE_DEVICES")
        os.system(command)
        os.system("nvidia-smi >>" + output_path + "/nvidia_smi_log.txt")
        print("\n\n\n")
        time.sleep(10)
