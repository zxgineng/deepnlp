import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config
from data_loader import load_pretrained_word_vec
from data_loader import load_pretrained_pos_vec


class Graph:
    def __init__(self, mode):
        self.mode = mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = True
        else:
            self.is_training = False

    def build(self, inputs):
        word_id = inputs['word_id']
        pos_id = inputs['pos_id']
        length = inputs['length']
        net = self._embedding(word_id, pos_id)
        net = self._bilstm(net, length)
        arc_head, arc_dep, label_head, label_dep = self._mlp(net)
        arc_logits = self._biaffine(arc_head, arc_dep, True, False, 1)
        label_logits = self._biaffine(label_head, label_dep, True, True, Config.model.label_num)
        return arc_logits, label_logits

    def _embedding(self, word_id, pos_id):
        wordvec = load_pretrained_word_vec()
        embedding = tf.get_variable('word_embedding', [wordvec.shape[0], wordvec.shape[1]],
                                    initializer=tf.constant_initializer(wordvec, tf.float32))
        word_embedding = tf.nn.embedding_lookup(embedding, word_id)

        posvec = load_pretrained_pos_vec()
        embedding = tf.get_variable('pos_embedding', [posvec.shape[0], posvec.shape[1]],
                                    initializer=tf.constant_initializer(posvec, tf.float32))
        pos_embedding = tf.nn.embedding_lookup(embedding, pos_id)
        embedding = tf.concat([word_embedding, pos_embedding], -1)
        outputs = slim.dropout(embedding, Config.model.embedding_keep_prob, is_training=self.is_training)
        return outputs

    def _bilstm(self, inputs, length):
        with tf.variable_scope('bilstm'):
            lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_size, initializer=tf.orthogonal_initializer)
            lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_size, initializer=tf.orthogonal_initializer)
            lstm_outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs, length,
                                                                          dtype=tf.float32)
            bilstm_outputs = tf.concat(lstm_outputs, axis=-1)
            outputs = slim.dropout(bilstm_outputs, Config.model.lstm_keep_prob, is_training=self.is_training)
            return outputs

    def _mlp(self, inputs):
        arc_head = slim.fully_connected(inputs, Config.model.arc_mlp_size)
        arc_head = slim.dropout(arc_head, Config.model.arc_keep_prob, is_training=self.is_training)
        arc_dep = slim.fully_connected(inputs, Config.model.arc_mlp_size)
        arc_dep = slim.dropout(arc_dep, Config.model.arc_keep_prob, is_training=self.is_training)

        label_head = slim.fully_connected(inputs, Config.model.label_mlp_size)
        label_head = slim.dropout(label_head, Config.model.label_keep_prob, is_training=self.is_training)
        label_dep = slim.fully_connected(inputs, Config.model.label_mlp_size)
        label_dep = slim.dropout(label_dep, Config.model.label_keep_prob, is_training=self.is_training)

        return arc_head, arc_dep, label_head, label_dep

    def _biaffine(self, head, dep, bias_head, bias_dep, out_channels):
        if bias_head:
            head = tf.concat([head, tf.ones_like(head)[:, :, 0:1]], -1)
        if bias_dep:
            dep = tf.concat([dep, tf.ones_like(dep)[:, :, 0:1]], -1)

        batch_num, seq_length, _ = head.shape
        dep_size = dep.shape.as_list()[-1]
        RU = tf.reshape(slim.fully_connected(head, dep_size * out_channels, None),
                        [batch_num, seq_length, dep_size, out_channels])
        RU = tf.transpose(RU, [0, 3, 1, 2])
        logits = tf.matmul(RU, tf.stack([tf.transpose(dep, [0, 2, 1])] * out_channels, 1))
        outputs = tf.squeeze(logits)
        return outputs
