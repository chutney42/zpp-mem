import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # hacked by Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from utils import *
from layer import *
from loader import *
from backpropagation import NeuralNetwork

file_name = "run_auto_increment"

class DFANeuralNetwork(NeuralNetwork):
    def build_optimize(self, output_vec):
        error = tf.subtract(output_vec, self.labels)
        self.step = []
        a = self.features
        for i, layer in enumerate(self.sequence):
            a = layer.build_optimize(a, error, i + 1 == len(self.sequence))
            if layer.trainable:
                self.step.append(layer.step)

if __name__ == '__main__':
    if not os.path.isfile(file_name):
        with open(file_name, 'w+') as file:
            file.write(str(0))

    training, test = load_mnist()
    DFA = DFANeuralNetwork(784,
                           [DFABlock([DFAFullyConnected(50), BatchNormalization(), Sigmoid()]),
                            DFABlock([DFAFullyConnected(30), BatchNormalization(), Sigmoid()]),
                            DFABlock([DFAFullyConnected(10), BatchNormalization(), Sigmoid()])],
                           10,
                           'DFA')
    DFA.build()
    DFA.train(training, test)
