import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config
from data_loader import load_pretrained_vec


class Graph:
    def __init__(self, mode):
        self.mode = mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = True
        else:
            self.is_training = False

    def build(self, inputs):
        word_id = inputs['word_id']
        char_image = inputs['char_image']
        length = inputs['length']
        word_embedding = self._embedding(word_id)
        cnn_embedding = self._cnn_embedding(char_image)
        net = tf.concat([word_embedding, cnn_embedding], -1)
        net = self._build_lstm(net, length)
        logits = self._build_fc(net)
        return logits

    def _embedding(self, inputs):
        wordvec = load_pretrained_vec()
        embedding = tf.get_variable('embedding', [wordvec.shape[0], wordvec.shape[1]],
                                    initializer=tf.constant_initializer(wordvec, tf.float32))
        output = tf.nn.embedding_lookup(embedding, inputs)
        return output

    def _cnn_embedding(self, inputs):
        net = slim.conv3d(inputs, 16, [1, 3, 3])
        net = slim.conv3d(net, 16, [1, 3, 3])
        net = slim.max_pool3d(net, [1, 3, 3], [1,2,2], 'SAME')
        net = slim.conv3d(net, 32, [1, 3, 3])
        net = slim.conv3d(net, 32, [1, 3, 3])
        net = slim.max_pool3d(net, [1, 3, 3], [1,2,2], 'SAME')
        net = tf.reshape(net, [-1, tf.shape(inputs)[1], tf.size(net[0][0])])
        outputs = slim.fully_connected(net, 100, tf.nn.tanh)
        return outputs

    def _build_lstm(self, inputs, length):
        with tf.variable_scope('bilstm'):
            lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer)
            lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer)
            lstm_outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs, length,
                                                                          dtype=tf.float32)
            outputs = tf.concat(lstm_outputs, axis=-1)

            return outputs

    def _build_fc(self, inputs):
        net = slim.dropout(inputs, Config.model.dropout_keep_prob, is_training=self.is_training)
        outputs = slim.fully_connected(net, Config.model.fc_unit, None)
        return outputs
