class single_parameter_generator():
    def __init__(self, parameter_name, range):
        self.parameter_name = parameter_name
        self.range = range

    def params(self, value=None):
        if value is None:
            for param_value in self.range:
                yield f"-{self.parameter_name} {param_value}" # TODO zmienic
        else:
            yield f"-{self.parameter_name} {value}"

