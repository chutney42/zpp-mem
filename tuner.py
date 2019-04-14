import os
import re

from hyperparameter_tuner.run_command_generator import run_command_generator as cmd_generator
from hyperparameter_tuner.single_parameter_generator import single_parameter_generator as sgen
from datetime import datetime
import time

result_regexp = re.compile(r'.*(total accuracy.*)\n')


def extract_to_csv(path):
    print(path)
    directory = os.fsencode(path)
    with open(f"{path}/summarise.csv", "w+") as output_file:
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if not filename.endswith(".csv"):
                with open(f"{path}/{filename}", "r") as input_file:
                    text = input_file.read()
                    matcher = result_regexp.match(text)
                    if matcher is not None:
                        result = f"file:{filename};result:{matcher.group(1)}"
                        output_file.write(result)
                        print(result)


if __name__ == '__main__':
    output_path = f"hyperparameter_tuner/results/{str(datetime.now()).replace(' ', '')}"
    vgg_16_BP_tuner = cmd_generator([sgen("sequence", ["fc1"]),
                                     sgen("propagator_initializer", ["he_normal", "he_uniform"]),
                                     sgen("name", ["first_dfa", "then_bp", "just_dfa", "just_bp"])],
                                    command_prefix="python experiment.py",
                                    output_path=output_path)\
        .run_commands()

    os.system(f"mkdir {output_path}")
    os.system(f"touch {output_path}/summarise.csv")

    for command in vgg_16_BP_tuner:
        print(command)
        os.system(command)
        os.system(f"rm {output_path}/summarise.csv")

        extract_to_csv(output_path)
        print("\n\n\n")
        time.sleep(10)
