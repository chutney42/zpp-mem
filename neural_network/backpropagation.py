from neural_network.backward_propagation import BackwardPropagation
from propagator.backward_propagator import Backpropagator


class Backpropagation(BackwardPropagation):
        def __init__(self, types, shapes, sequence, cost_function_name, propagator_initializer=None, *args, **kwargs):
            super().__init__(types, shapes, sequence, cost_function_name, Backpropagator(), *args, **kwargs)
