from layer.layer import Layer


class WeightLayer(Layer):
    def __init__(self, learning_rate=None, momentum=None, scope="weight_layer"):
        super().__init__(trainable=True, scope=scope)
        self.propagator = None
        self.learning_rate = learning_rate
        self.momentum = momentum

    def build_propagate(self, error, gather_stats=False):
        raise NotImplementedError("This method should be implemented in subclass")

    def build_update(self, error, gather_stats=False):
        raise NotImplementedError("This method should be implemented in subclass")

    def build_backward(self, error, gather_stats=False):
        if not self.propagator:
            raise AttributeError("The propagator should be specified")
        propagated_error = self.build_propagate(error, gather_stats)
        self.build_update(error, gather_stats)
        return propagated_error
