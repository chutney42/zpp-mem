from itertools import product

from hyperparameter_tuner.single_parameter_generator.single_parameter_generator import \
    single_parameter_generator as sgen


class run_command_generator():
    def __init__(self, single_parameter_generator_list, command_prefix="python ../experiment.py",
                 output_path="./results"):
        for gen in single_parameter_generator_list:
            assert isinstance(gen, sgen)
        self.single_parameter_generator_list = single_parameter_generator_list
        self.run_command = command_prefix
        self.output_path = output_path

    def run_commands(self):
        all_parrams_gennerator = self.single_parameter_generator_list[0].params()
        for p in self.single_parameter_generator_list[1:]:
            all_parrams_gennerator = product(all_parrams_gennerator, p.params())
        for train_params in all_parrams_gennerator:
            command = str(train_params).replace('(', '').replace(')', '').replace('\'', '').replace(',', '')
            command = self.run_command + " " + command + " > " + self.output_path + "/" + command.replace(' ',
                                                                                                          '_').replace(
                '-', '').replace('.', '')

            yield command


def default_commands_generator(command_prefix="python experiment.py", output_path="./hyperparameter_tuner/results"):
    return run_command_generator([sgen("name", ["vgg_16"]),
                                  sgen("learning_rate", [0.001, 0.005, 0.01, 0.03, 0.07, 0.1, 0.5, 1]),
                                  sgen("batch_size", [20, 25, 30, 35, 50, 75]),
                                  ], command_prefix=command_prefix, output_path=output_path).run_commands()


if __name__ == '__main__':
    commands = default_commands_generator()
    for c in commands:
        print(c)
