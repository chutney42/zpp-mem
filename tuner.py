import os

from hyperparameter_tuner.run_command_generator import run_command_generator
from hyperparameter_tuner.single_parameter_generator import \
    single_parameter_generator as sgen
from datetime import datetime
import time


def extract_to_csv(path):
    print(path)
    directory = os.fsencode(path)
    output_file = open(f"{path}/summarise.csv", "w+")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if not filename.endswith(".csv"):
            input_file = open(f"{path}/{filename}", "r")
            lines = input_file.readlines()
            output_file.write(f"file:{filename};result:{lines[-2]}")
            print(f"file:{filename};result:{lines[-2]}")
            input_file.close()
    output_file.close()


if __name__ == '__main__':
    output_path = f"hyperparameter_tuner/results/{str(datetime.now()).replace(' ', '')}"
    vgg_16_BP_tuner = run_command_generator([sgen("name", ["vgg_16"]),
                                             sgen("learning_rate", [0.001, 0.005, 0.01, 0.03, 0.07, 0.1, 0.5, 1]),
                                             sgen("batch_size", [20, 25, 30, 35, 50, 75]),
                                             ], command_prefix="python experiment.py",
                                            output_path=output_path).run_commands()

    os.system(f"mkdir {output_path}")
    for command in vgg_16_BP_tuner:
        print(command)
        os.system(command)
        print("\n\n\n")
        time.sleep(10)
    extract_to_csv(output_path)
