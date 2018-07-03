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
        word = inputs['id']
        predicate = inputs['pid']
        length = inputs['length']
        embeddings = self.embedding(word,predicate)
        net = self.build_highway_lstm(embeddings,length,reverse=False,highway=False,name='hw_lstm1')
        for n in range(Config.model.num_lstm_layer-1):
            num = (n+1)%2
            net = self.build_highway_lstm(net,length,reverse=bool(num),highway=True,name='hw_lstm' + str(n+2))
        logits = slim.fully_connected(net,Config.model.num_class,activation_fn=None)
        return logits

    def embedding(self, word,predicate):
        wordvec = load_pkl(Config.data.wordvec_file)
        with tf.variable_scope('embedding'):
            word_embedding = tf.get_variable('word_embedding', [wordvec.shape[0], wordvec.shape[1]],
                                             initializer=tf.constant_initializer(wordvec, tf.float32))
            word = tf.nn.embedding_lookup(word_embedding, word)
            predicate = tf.nn.embedding_lookup(word_embedding, predicate)
            embeddings = tf.concat([word,predicate],-1)
            return embeddings

    def build_highway_lstm(self, inputs, length, reverse=None,highway=None, name=None):
        with tf.variable_scope(name):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer)
            if self.is_training:
                # variational dropout
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,state_keep_prob=Config.train.keep_prob,variational_recurrent=True,dtype=tf.float32)
            if reverse:
                reverse_inputs = tf.reverse_sequence(inputs,length,1)
                lstm_outputs,_ = tf.nn.dynamic_rnn(lstm_cell,reverse_inputs,length,dtype=tf.float32)
                outputs = tf.reverse_sequence(lstm_outputs,length,1)
            else:
                outputs,_ = tf.nn.dynamic_rnn(lstm_cell,inputs,length,dtype=tf.float32)
            if highway:
                outputs_prev = tf.concat([outputs[:,-1:,:] * 0.0, outputs[:,:-1,:]],1)
                transfer_gate = slim.fully_connected(tf.concat([outputs_prev,inputs],-1),Config.model.lstm_unit,tf.nn.sigmoid)
                carry_gate = 1.0 - transfer_gate
                outputs = outputs * transfer_gate + inputs * carry_gate
            return outputs

