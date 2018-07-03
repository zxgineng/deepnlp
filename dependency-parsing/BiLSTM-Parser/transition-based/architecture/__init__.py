from utils import Config
import tensorflow as tf
from tensorflow.contrib import slim

from data_loader import load_pkl


class Graph:
    def __init__(self, mode):
        self.mode = mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = True
        else:
            self.is_training = False

    def build(self, inputs):
        word = inputs['word']
        pos = inputs['pos']
        length = inputs['length']
        embeddings = self.embedding(word,pos)
        lstm_ouputs = self.build_lstm(embeddings,length)
        mlp_func = self.build_mlp_func()
        return lstm_ouputs, mlp_func

    def embedding(self, word,pos):
        wordvec = load_pkl(Config.data.wordvec_file)
        posvec = load_pkl(Config.data.posvec_file)
        with tf.variable_scope('embedding'):
            word_embedding = tf.get_variable('word_embedding', [wordvec.shape[0], wordvec.shape[1]],
                                             initializer=tf.constant_initializer(wordvec, tf.float32))
            pos_embedding = tf.get_variable('pos_embedding', [posvec.shape[0], posvec.shape[1]],
                                            initializer=tf.constant_initializer(posvec, tf.float32))
            word = tf.nn.embedding_lookup(word_embedding, word)
            pos = tf.nn.embedding_lookup(pos_embedding, pos)
            embeddings = tf.concat([word,pos],-1)

            return embeddings

    def build_lstm(self, inputs, length):
        with tf.variable_scope('bilstm'):
            lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer)
            lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer)
            lstm_outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs, length,
                                                                          dtype=tf.float32)
            lstm_outputs = tf.concat(lstm_outputs, axis=-1)
            return lstm_outputs

    def build_mlp_func(self):
        fc1_weights = tf.get_variable('fc1/weights',[Config.model.lstm_unit * 4, Config.model.fc_unit],tf.float32)
        fc1_biases = tf.get_variable('fc1/biases',[Config.model.fc_unit],tf.float32)
        fc2_weights = tf.get_variable('fc2/weights',[Config.model.fc_unit,Config.model.num_class],tf.float32)
        fc2_biases = tf.get_variable('fc2/biases',[Config.model.num_class],tf.float32)

        def mlp(inputs,activation = tf.nn.tanh):
            outputs = tf.matmul(activation(tf.matmul(inputs,fc1_weights) + fc1_biases),fc2_weights) + fc2_biases
            return outputs
        return mlp