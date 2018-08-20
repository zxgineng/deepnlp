from utils import Config
import tensorflow as tf
from tensorflow.contrib import slim

from data_loader import load_pretrained_vec


class Graph(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Graph, self).__init__(**kwargs)

        self.embedding_layer = Embedding_Layer(name='embedding_layer')
        self.encoding_layer = Encoding_layer(name='encoding_layer')
        self.softmax_dense = tf.keras.layers.Dense(Config.model.class_num, name='softmax_dense')

    def call(self, inputs, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            training = True
        else:
            training = False

        word_id = inputs['word_id']
        tag_id = inputs['tag_id']
        predicate = inputs['predicate']
        length = inputs['length']

        embedded = self.embedding_layer(word_id, tag_id, predicate)
        encoded = self.encoding_layer(embedded, length, training)
        logits = self.softmax_dense(encoded)

        return logits


class Embedding_Layer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Embedding_Layer, self).__init__(**kwargs)

        wordvec = load_pretrained_vec()
        self.word_embedding = tf.keras.layers.Embedding(wordvec.shape[0], wordvec.shape[1],
                                                        tf.constant_initializer(wordvec, tf.float32),
                                                        name='word_embedding')
        self.tag_embedding = tf.keras.layers.Embedding(Config.model.tag_num, Config.model.tag_embedding_size,
                                                       name='tag_embedding')

    def call(self, word_id, tag_id, predicate):
        word_embedded = self.word_embedding(word_id)
        tag_embedded = self.tag_embedding(tag_id)
        predicate = tf.cast(tf.tile(tf.expand_dims(predicate, -1), [1, 1, Config.model.predicate_size]), tf.float32)
        outputs = tf.concat([word_embedded, tag_embedded, predicate], -1)
        return outputs


class Encoding_layer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Encoding_layer, self).__init__(**kwargs)

    def _highway_bilstm(self, inputs, length, training, reverse=None, highway=None, name=None):
        with tf.variable_scope(name):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer)
            if training:
                # variational dropout
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, state_keep_prob=Config.train.keep_prob,
                                                          variational_recurrent=True, dtype=tf.float32)
            if reverse:
                reverse_inputs = tf.reverse_sequence(inputs, length, 1)
                lstm_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, reverse_inputs, length, dtype=tf.float32)
                outputs = tf.reverse_sequence(lstm_outputs, length, 1)
            else:
                outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs, length, dtype=tf.float32)
            if highway:
                prev_h = tf.concat([outputs[:, -1:, :] * 0.0, outputs[:, :-1, :]], 1)  # h[t-1]
                transfer_gate = slim.fully_connected(tf.concat([prev_h, inputs], -1), Config.model.lstm_unit,
                                                     tf.nn.sigmoid)
                carry_gate = 1.0 - transfer_gate
                outputs = outputs * transfer_gate + \
                          slim.fully_connected(inputs, Config.model.lstm_unit, None,
                                               biases_initializer=None) * carry_gate
            return outputs

    def call(self, inputs, length, training):
        net = self._highway_bilstm(inputs, length, training=training, reverse=False, highway=False, name='hw_lstm1')
        for n in range(Config.model.num_lstm_layer - 1):
            num = (n + 1) % 2
            net = self._highway_bilstm(net, length, training=training, reverse=bool(num), highway=True,
                                       name='hw_lstm' + str(n + 2))
        return net
