import os
import re
import subprocess
from subprocess import Popen

from hyperparameter_tuner.run_command_generator import default_commands_generator
# total accuracy: 69.23999786376953% iterations: 16000
# learning process took 63.48976540565491 seconds (realtime)

if __name__ == '__main__':
    for command in default_commands_generator():
        print(command)
        os.system(command)

