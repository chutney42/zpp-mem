import tensorflow as tf

def backpropagation(error, weights, name=None):
    with tf.name_scope(name or "backpropagation"):
        return tf.matmul(error, tf.transpose(weights))

def feedback_alignment(error, weights, name=None):
    with tf.name_scope(name or "feedback_alignment"):
        random_weights = tf.get_variable("random_weights", shape=tf.transpose(weights).get_shape().as_list(),
            initializer=tf.random_normal_initializer())
        return tf.matmul(error, random_weights)

def direct_feedback_alignment(error, weights, name=None):
    with tf.name_scope(name or "direct_feedback_alignment"):
        random_weights = tf.get_variable("random_weights", shape=[error.shape[1], weights.shape[0]],
            initializer=tf.random_normal_initializer())
        return tf.matmul(error, random_weights)
