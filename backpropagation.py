from backward_propagation import BackwardPropagation
from propagate import backpropagation


class Backpropagation(BackwardPropagation):
    def set_propagate_functions(self):
        for block in self.sequence:
            block.head.propagate_func = backpropagation
