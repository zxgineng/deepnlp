from utils import Config
import tensorflow as tf
from tensorflow.contrib import slim

from data_loader import read_pretrained_wordvec


class Graph:
    def __init__(self, mode):
        self.mode = mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = True
        else:
            self.is_training = False

    def build(self, inputs):
        id = inputs['id']
        length = inputs['length']
        embeddings = self.embedding(id)
        logits = self.build_lstm(embeddings, length)
        return logits

    def embedding(self, inputs):
        _, wordvec = read_pretrained_wordvec(Config.data.wordvec_fname)
        embedding = tf.get_variable('embedding', [wordvec.shape[0], wordvec.shape[1]],
                                    initializer=tf.constant_initializer(wordvec, tf.float32), trainable=False)
        output = tf.nn.embedding_lookup(embedding, inputs)
        output = slim.dropout(output, Config.model.dropout_keep_prob, is_training=self.is_training)
        return output

    def build_lstm(self, inputs, length):
        with tf.variable_scope('bilstm'):
            lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer)
            lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer)
            lstm_outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs, length,
                                                                          dtype=tf.float32)
            lstm_outputs = tf.concat(lstm_outputs, axis=-1)
            outputs = slim.fully_connected(lstm_outputs, Config.model.fc_unit, activation_fn=None)
            return outputs
