from util.loader import *
from neural_network.neural_network import DirectFeedbackAlignment
from propagator.direct_propagator import DirectFixedRandom, DirectPropagator


class DirectFeedbackAlignment(DirectFeedbackAlignment):
    def __init__(self, types, shapes, sequence, propagator=None, *args, **kwargs):
        if not propagator:
            propagator = DirectFixedRandom(shapes[1][0].value)
        elif not isinstance(propagator, DirectPropagator):
            raise TypeError("propagator for DirectFeedbackAlignment must be instance of DirectPropagator")
        super().__init__(types, shapes, sequence, propagator, *args, **kwargs)

    def build_forward(self):
        a = self.features
        for block in self.sequence:
            for layer in block:
                a = layer.build_forward(a, remember_input=False)
        return a

    def build_backward(self, output_vec):
        output_error = tf.subtract(output_vec, self.labels)
        self.step = []
        a = self.features
        for i, block in enumerate(self.sequence):
            for layer in block:
                a = layer.build_forward(a, remember_input=True)
            if i + 1 < len(self.sequence):
                error = self.sequence[i + 1].head.build_propagate(output_error)
            else:
                error = output_error
            for layer in reversed(block.tail):
                error = layer.build_backward(error)
                if layer.trainable:
                    self.step.append(layer.step)
            block.head.build_update(error)
            self.step.append(block.head.step)
