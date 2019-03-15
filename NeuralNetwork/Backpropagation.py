from NeuralNetwork.BackwardPropagation import BackwardPropagation
from Propagators.BackwardPropagator import Backpropagator


class Backpropagation(BackwardPropagation):
        def __init__(self, types,shapes , sequence, propagator=Backpropagator(), *args, **kwargs):
            super().__init__(types,shapes, sequence, propagator, *args, **kwargs)
