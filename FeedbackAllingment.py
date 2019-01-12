import backpropagation
import tensorflow as tf


class FANeuralNetwork(backpropagation.NeuralNetwork):
    # def __init__(self, sizes, eta=0.5):
    #     super().__init__(self,sizes,eta=eta)

    def __backpropagation(self, da, z, a, eta, scope):
        with tf.variable_scope(scope, reuse=True) as scope:
            # FAw = tf.get_variable("backprop_FAweights")
            w = tf.get_variable("weights")
            FAw = tf.get_variable("backprop_FAweights", shape=tf.transpose(w).get_shape().as_list(),
                                  initializer=tf.zeros_initializer())
            b = tf.get_variable("biases")
            dz = tf.multiply(da, backpropagation.sigmoid_prime(z))
            db = dz
            dw = tf.matmul(tf.transpose(a), dz)
            dal = tf.matmul(dz, FAw)
            return dal, \
                   tf.assign(w, tf.subtract(w, tf.multiply(eta, dw))), \
                   tf.assign(b, tf.subtract(b, tf.multiply(eta, tf.reduce_mean(db, axis=[0]))))


if __name__ == '__main__':
    NN = FANeuralNetwork([784, 50, 30, 10])
    NN.load_data()
    NN.train()
