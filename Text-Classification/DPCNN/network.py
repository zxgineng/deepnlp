from utils import Config
import tensorflow as tf
from tensorflow.contrib import slim

from data_loader import load_pretrained_vec


class Graph:
    def __init__(self, mode):
        self.mode = mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = True
        else:
            self.is_training = False

    def build(self, inputs):
        net = self._wordvec_embedding(inputs)
        net = self._region_embedding(net)
        net = self._change_dim_conv(net)

        conv_block_num = 0
        while net.shape[1] > 1 and conv_block_num <= Config.model.max_conv_block_num:
            net = self._conv_block(net)
            conv_block_num += 1

        logits = self._fc_layer(net)
        return logits

    def _wordvec_embedding(self, inputs):
        wordvec = load_pretrained_vec()
        embedding = tf.get_variable('embedding', [wordvec.shape[0], wordvec.shape[1]],
                                    initializer=tf.constant_initializer(wordvec, tf.float32))
        output = tf.nn.embedding_lookup(embedding, inputs)
        return output

    def _region_embedding(self, inputs):
        outputs = []
        for kernel_size in Config.model.kernel_sizes:
            conv = tf.layers.conv1d(inputs, Config.model.fixed_channel, kernel_size, 1, 'SAME',
                                    kernel_initializer=tf.random_normal_initializer(0.0, 0.01),
                                    kernel_regularizer=slim.l2_regularizer(Config.train.regular_weight))
            outputs.append(conv)
        outputs = tf.concat(outputs, -1)
        return outputs

    def _change_dim_conv(self, inputs):
        return tf.layers.conv1d(tf.nn.relu(inputs), Config.model.fixed_channel, 1, 1, 'SAME',
                                kernel_initializer=tf.random_normal_initializer(0.0, 0.01),
                                kernel_regularizer=slim.l2_regularizer(Config.train.regular_weight))

    def _conv_block(self, inputs):
        net = tf.layers.conv1d(tf.nn.relu(inputs), Config.model.fixed_channel, 3, 1, 'SAME',
                               kernel_initializer=tf.random_normal_initializer(0.0, 0.01),
                               kernel_regularizer=slim.l2_regularizer(Config.train.regular_weight))
        net = tf.layers.conv1d(tf.nn.relu(net), Config.model.fixed_channel, 3, 1, 'SAME',
                               kernel_initializer=tf.random_normal_initializer(0.0, 0.01),
                               kernel_regularizer=slim.l2_regularizer(Config.train.regular_weight))
        net = net + inputs
        outputs = tf.layers.max_pooling1d(net, 3, 2, 'SAME')
        return outputs

    def _fc_layer(self, inputs):
        net = slim.flatten(inputs)
        net = slim.dropout(net, Config.train.dropout_keep_prob, is_training=self.is_training)
        outputs = slim.fully_connected(net, Config.model.fc_unit, None,
                                       weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                       weights_regularizer=slim.l2_regularizer(Config.train.regular_weight))
        return outputs
