import  backpropagation
import tensorflow as tf


class FANeuralNetwork(backpropagation.NeuralNetwork):
    def __init__(self, sizes, eta=0.5):

        self.back_FAlayers = []
        for i in range(1, len(sizes)):
            w = tf.get_variable("backprop_FAweights", [sizes[i], sizes[i - 1]],
                                initializer=tf.random_normal_initializer())
            self.back_FAlayers.append(w)

        super().init()

    def __backpropagation(self, da, z, a, eta, scope):
        with tf.variable_scope(scope, reuse=True) as scope:
            # FAw = tf.get_variable("backprop_FAweights")
            FAw =
            w = tf.get_variable("weights")
            b = tf.get_variable("biases")
            dz = tf.multiply(da, backpropagation.sigmoid_prime(z))
            db = dz
            dw = tf.matmul(tf.transpose(a), dz)
            dal = tf.matmul(dz, FAw)
            return dal, tf.assign(w, tf.subtract(w, tf.multiply(eta, dw))), tf.assign(b, tf.subtract(b, tf.multiply(eta,
                                                                                                                    tf.reduce_mean(


                                                                                                                        db,
                                                                                                                        axis=[
                                                                                                                            0]))))



if __name__ == '__main__':
    NN = FANeuralNetwork([784, 50, 30, 10])
    NN.load_data()
    NN.train()
