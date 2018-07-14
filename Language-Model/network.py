import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config


class Graph:
    def __init__(self, mode):
        self.mode = mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = True
        else:
            self.is_training = False

    def build(self, inputs):
        for_inputs = inputs['for_inputs']
        back_inputs = inputs['back_inputs']
        for_embeddings = self._cnn_embedding(for_inputs)
        back_embeddings = self._cnn_embedding(back_inputs)
        for_embeddings = self._highway(for_embeddings)
        back_embeddings = self._highway(back_embeddings)
        with tf.variable_scope('for_lstm'):
            for_outputs = self._lstm(for_embeddings)
        with tf.variable_scope('back_lstm'):
            back_outputs = self._lstm(back_embeddings)
        return for_outputs, back_outputs

    def _cnn_embedding(self, inputs):
        """inputs: [b,s,w,h,c]  outputs: [b*s,w,h,c]"""
        with tf.variable_scope('cnn_embedding', reuse=tf.AUTO_REUSE):
            net = tf.reshape(inputs, [-1, Config.model.char_image_size, Config.model.char_image_size, 1])
            net = slim.conv2d(net, 16, 3)
            net = slim.conv2d(net, 16, 3)
            net = slim.max_pool2d(net, 3, 2, 'SAME')
            net = slim.conv2d(net, 32, 3)
            net = slim.conv2d(net, 32, 3)
            outputs = slim.max_pool2d(net, 3, 2, 'SAME')
            return outputs

    def _highway(self, inputs):
        """inputs: [b*s,w,h,c]  outputs: [b,s,c]"""
        with tf.variable_scope('highway', reuse=tf.AUTO_REUSE):
            net = slim.flatten(inputs)
            num = net.shape.as_list()[-1]
            for _ in range(Config.model.highway_layer_num):
                carry_gate = slim.fully_connected(net, num, tf.nn.sigmoid)
                transform_gate = slim.fully_connected(net, num)
                net = carry_gate * transform_gate + (1.0 - carry_gate) * net
            net = slim.fully_connected(net, Config.model.embedding_size, None)
            outputs = tf.reshape(net, [-1, Config.model.seq_length, Config.model.embedding_size])
            return outputs

    def _lstm(self, inputs):
        """inputs: [b,s,c]  outputs: [b*s,c]"""
        lstm_cells = []
        for i in range(Config.model.lstm_layer_num):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, num_proj=Config.model.embedding_size,
                                                cell_clip=Config.model.lstm_cell_clip,
                                                proj_clip=Config.model.lstm_proj_clip)
            if i != 0:  # don't add skip connection from token embedding to 1st layer output
                lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)
            if self.is_training:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=Config.model.dropout_keep_prob)
            lstm_cells.append(lstm_cell)

            # inner layer embedding, no run while training
            inner_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
            outputs, _ = tf.nn.dynamic_rnn(inner_lstm, inputs, dtype=tf.float32)

        # if Config.model.lstm_layer_num > 1:
        #     lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
        # else:
        #     lstm_cell = lstm_cells[0]
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
        tf.identity(outputs, str(Config.model.lstm_layer_num))  # final layer embedding
        outputs = tf.reshape(outputs, [-1, Config.model.embedding_size])
        outputs = slim.dropout(outputs, Config.model.dropout_keep_prob, is_training=self.is_training)
        return outputs

