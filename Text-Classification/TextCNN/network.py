from utils import Config
import tensorflow as tf
from tensorflow.contrib import slim

from data_loader import load_pretrained_vec


class Graph:
    def __init__(self, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = True
        else:
            self.is_training = False

    def build(self, inputs):

        net = self._wordvec_embedding(inputs)

        net = self._build_conv_maxpool_layer(net)
        logits = self._fc_layer(net)
        return logits

    def _wordvec_embedding(self, inputs):
        wordvec = load_pretrained_vec()
        embedding = tf.get_variable('embedding', [wordvec.shape[0], wordvec.shape[1]],
                                    initializer=tf.constant_initializer(wordvec, tf.float32))
        output = tf.nn.embedding_lookup(embedding, inputs)
        return output

    def _build_conv_maxpool_layer(self, inputs):
        outputs = []
        for kernel_size in Config.model.kernel_sizes:
            conv = tf.layers.conv1d(inputs, Config.model.num_filters, kernel_size, 1, 'SAME',
                                    activation=tf.nn.relu, kernel_initializer=slim.xavier_initializer())
            pool = tf.reduce_max(conv, 1)
            outputs.append(pool)
        outputs = tf.concat(outputs, 1)

        return outputs

    def _fc_layer(self, inputs):
        net = slim.dropout(inputs, Config.model.dropout_keep_prob, is_training=self.is_training)
        outputs = slim.fully_connected(net, Config.model.fc_unit, None)
        return outputs
