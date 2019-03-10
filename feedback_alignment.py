from backward_propagation import BackwardPropagation
from propagate import feedback_alignment

class FeedbackAlignment(BackwardPropagation):
    def set_propagate_functions(self):
        for block in self.sequence:
            block.head.propagate_func = feedback_alignment

