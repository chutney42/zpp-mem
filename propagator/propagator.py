class Propagator(object):
    def propagate_conv(self, layer, error):
        raise NotImplementedError("This method should be implemented in subclass")

    def propagate_fc(self, layer, error):
        raise NotImplementedError("This method should be implemented in subclass")

    def get_filter(self, filters):
        raise NotImplementedError("This method should be implemented in subclass")

    def get_weights(self, weights):
        raise NotImplementedError("This method should be implemented in subclass")


