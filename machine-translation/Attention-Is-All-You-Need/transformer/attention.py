import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


__all__ = [
    "positional_encoding", "Attention"
]


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    """create position embedding matrix, shape:[max_seq_length,model_dim]"""
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


class Attention:
    """Attention class"""

    def __init__(self,
                 num_heads=1,
                 masked=False,
                 linear_key_dim=50,
                 linear_value_dim=50,
                 model_dim=100,
                 dropout=0.2):

        assert linear_key_dim % num_heads == 0
        assert linear_value_dim % num_heads == 0

        self.num_heads = num_heads
        self.masked = masked
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.dropout = dropout

    def multi_head(self, q, k, v):
        """multi-head atttention"""
        q, k, v = self._linear_projection(q, k, v)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs)
        output = self._concat_heads(outputs)
        output = slim.fully_connected(output,self.model_dim,activation_fn=None)

        return tf.nn.dropout(output, 1.0 - self.dropout)

    def _linear_projection(self, q, k, v):
        # shape:[batch_size,max_seq_length,linear_key_dim]
        q = slim.fully_connected(q,self.linear_key_dim,activation_fn=None,biases_initializer=None)
        k = slim.fully_connected(k,self.linear_key_dim,activation_fn=None,biases_initializer=None)
        # shape:[batch_size,max_seq_length,linear_value_dim]
        v = slim.fully_connected(v,self.linear_value_dim,activation_fn=None,biases_initializer=None)
        return q, k, v

    def _split_heads(self, q, k, v):

        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            tensor = tf.reshape(tensor, [-1, tensor.shape[1], num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3])
        # shape:[batch_size, num_heads, max_seq_len,dim // num_heads]
        qs = split_last_dimension_then_transpose(q, self.num_heads, self.linear_key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.linear_key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.linear_value_dim)

        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs):
        key_dim_per_head = self.linear_key_dim // self.num_heads
        # shape:[batch_size, num_heads, max_seq_len,max_seq_len]
        o1 = tf.matmul(qs, ks, transpose_b=True)
        # scale
        o2 = o1 / (key_dim_per_head**0.5)
        # mask in decoder network
        if self.masked:
            diag_vals = tf.ones_like(o2[0, 0, :, :])
            # create lower triangular
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            # tile to shape:[batch_size, num_heads, max_seq_len,max_seq_len]
            masks = tf.tile(tf.reshape(tril, [1, 1] + tril.get_shape().as_list()),
                            [tf.shape(o2)[0], tf.shape(o2)[1], 1, 1])
            paddings = tf.ones_like(masks) * -1e9
            o2 = tf.where(tf.equal(masks, 0), paddings, o2)

        o3 = tf.nn.softmax(o2)
        return tf.matmul(o3, vs)

    def _concat_heads(self, outputs):

        def transpose_then_concat_last_two_dimenstion(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3])
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])
        # shape:[batch_size,max_seq_len,linear_key_dim]
        return transpose_then_concat_last_two_dimenstion(outputs)
