from neural_network.backward_propagation import BackwardPropagation
from propagator.backward_propagator import Backpropagator


class Backpropagation(BackwardPropagation):
        def __init__(self, types,shapes , sequence, propagator=Backpropagator(), *args, **kwargs):
            super().__init__(types,shapes, sequence, propagator, *args, **kwargs)
