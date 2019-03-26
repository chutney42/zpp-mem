class CostFunction(object):
    @staticmethod
    def cost(output, labels, scope="cost_function"):
        raise NotImplementedError("This method should be implemented in subclass")

    @staticmethod
    def error(predictions, labels, scope="cost_function"):
        raise NotImplementedError("This method should be implemented in subclass")
