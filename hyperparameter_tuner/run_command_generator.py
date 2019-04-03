from itertools import product

import hyperparameter_tuner.single_parameter_generator.single_parameter_generator  as gens


class run_command_generator():
    def __init__(self, single_parameter_generator_list, run_command="python ../experiment.py", output_path="./results"):
        for gen in single_parameter_generator_list:
            assert isinstance(gen, gens.single_parameter_generator)
        self.single_parameter_generator_list = single_parameter_generator_list
        self.run_command = run_command
        self.output_path = output_path

    def run_commands(self):
        all_parrams_gennerator = self.single_parameter_generator_list[0].params()
        for p in self.single_parameter_generator_list[1:]:
            all_parrams_gennerator = product(all_parrams_gennerator, p.params())
        for train_params in all_parrams_gennerator:
            command = str(train_params).replace('(', '').replace(')', '').replace('\'', '').replace(',', '')
            command = self.run_command + " " + command + " > " + self.output_path + "/" + command.replace(' ',
                                                                                                          '_').replace(
                '-', '')

            yield command


def default_commands_generator():
    return run_command_generator([gens.batch_size(),
                                  gens.learning_type(),
                                  gens.name(),
                                  # gens.cost_function(),
                                  # gens.sequence(),
                                  gens.learning_rate()]).run_commands()


if __name__ == '__main__':
    commands = default_commands_generator()
    for c in commands:
        print(c)
