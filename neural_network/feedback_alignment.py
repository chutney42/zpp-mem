import tensorflow as tf

from neural_network.backward_propagation import BackwardPropagation
from propagator.backward_propagator import FixedRandom


class FeedbackAlignment(BackwardPropagation):
    def __init__(self, types, shapes, sequence, cost_function_name,
                 propagator_initializer=tf.random_normal_initializer(), *args, **kwargs):
            propagator = FixedRandom(propagator_initializer)
            super().__init__(types, shapes, sequence, cost_function_name, propagator, *args, **kwargs)
