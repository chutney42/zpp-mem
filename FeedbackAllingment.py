from backpropagation import NeuralNetwork

class FANeuralNetwork(NeuralNetwork):
    def __init__(self, sizes, eta=0.5):
        super().init()
        self.back_FAlayers =[]
        for i in range(1,self.num_layers):
            self.back_FAlayers.app
    def __backpropagation(self, da, z, a, eta, scope):
        pass
