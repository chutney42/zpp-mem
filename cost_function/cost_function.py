class CostFunction(object):
    @staticmethod
    def cost(output, labels, name=None):
        raise NotImplementedError("This method should be implemented in subclass")

    @staticmethod
    def error(predictions, labels, name=None):
        raise NotImplementedError("This method should be implemented in subclass")
