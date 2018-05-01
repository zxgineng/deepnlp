import tensorflow as tf
from tensorflow.contrib import slim


class FFN():
    """FFN class (Position-wise Feed-Forward Networks)"""

    def __init__(self,
                 w1_dim=200,
                 w2_dim=100,
                 dropout=0.1):

        self.w1_dim = w1_dim
        self.w2_dim = w2_dim
        self.dropout = dropout

    def dense_relu_dense(self, inputs):
        # output = tf.layers.dense(inputs, self.w1_dim, activation=tf.nn.relu)
        output = slim.fully_connected(inputs,self.w1_dim)
        # output =tf.layers.dense(output, self.w2_dim)
        output = slim.fully_connected(output,self.w2_dim,activation_fn=None)

        return tf.nn.dropout(output, 1.0 - self.dropout)

    # def conv_relu_conv(self):
    #     raise NotImplementedError("i will implement it!")
