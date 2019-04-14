from layer.layer import Layer


class WeightLayer(Layer):
    def __init__(self, func=None, scope="weight_layer"):
        super().__init__(trainable=True, scope=scope)
        self.func = func
