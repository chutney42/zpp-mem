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

if __name__ == '__main__':
    if not os.path.isfile(file_name):
        with open(file_name, 'w+') as file:
            file.write(str(0))

    training, test = load_mnist()
    FA = NeuralNetwork(784,
                       [FAFullyConnected(50),
                        Sigmoid(),
                        FAFullyConnected(30),
                        Sigmoid(),
                        FAFullyConnected(10),
                        Sigmoid()],
                       10,
                       'FA')
    FA.build()
    FA.train(training, test)
