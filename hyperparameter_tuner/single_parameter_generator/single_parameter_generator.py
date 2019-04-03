class single_parameter_generator():
    def __init__(self, parameter_name, range):
        self.parameter_name = parameter_name
        self.range = range

    def params(self, value=None):
        if value is None:
            for param_value in self.range:
                yield f"-{self.parameter_name} {param_value}"
        else:
            yield f"-{self.parameter_name} {value}"


class learning_type(single_parameter_generator):
    def __init__(self):
        super().__init__("learning_type", ["DFA"])


class learning_rate(single_parameter_generator):
    def __init__(self):
        super().__init__("learning_rate", [0.001, 0.005, 0.01, 0.03, 0.07])


class batch_size(single_parameter_generator):
    def __init__(self):
        super().__init__("batch_size", [20, 25, 30, 35, 50, 75])


class cost_function(single_parameter_generator):
    def __init__(self):
        super().__init__("cost_function", ["mean_squared_error"])


class sequence(single_parameter_generator):
    def __init__(self):
        super().__init__("sequence", ["fc1"])


class name(single_parameter_generator):
    def __init__(self):
        super().__init__("name", ["vgg_16"])
