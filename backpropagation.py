import tensorflow as tf
from backward_propagation import BackwardPropagation
from propagator import Backpropagator

class Backpropagation(BackwardPropagation):
        def __init__(self, input_dim, sequence, output_dim, propagator=Backpropagator(), *args, **kwargs):
            super().__init__(input_dim, sequence, output_dim, propagator, *args, **kwargs)
