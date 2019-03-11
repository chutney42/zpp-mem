from backward_propagation import BackwardPropagation
from propagator import FixedRandom


class FeedbackAlignment(BackwardPropagation):
    def __init__(self, input_dim, sequence, output_dim, propagator=FixedRandom(), *args, **kwargs):
            super().__init__(input_dim, sequence, output_dim, propagator, *args, **kwargs)
