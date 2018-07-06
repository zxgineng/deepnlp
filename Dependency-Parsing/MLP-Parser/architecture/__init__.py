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
        embeddings = self.embedding(inputs)
        net = self.cube_activation(embeddings)
        logits = self.fc_layer(net)
        return logits

    def embedding(self, inputs):
        word = inputs['word']
        pos = inputs['pos']
        dep = inputs['dep']
        wordvec = load_pkl(Config.data.wordvec_file)
        posvec = load_pkl(Config.data.posvec_file)
        depvec = load_pkl(Config.data.depvec_file)
        with tf.variable_scope('embedding'):
            word_embedding = tf.get_variable('word_embedding', [wordvec.shape[0], wordvec.shape[1]],
                                             initializer=tf.constant_initializer(wordvec, tf.float32),
                                             regularizer=slim.l2_regularizer(Config.model.reg_scale))
            pos_embedding = tf.get_variable('pos_embedding', [posvec.shape[0], posvec.shape[1]],
                                            initializer=tf.constant_initializer(posvec, tf.float32),
                                            regularizer=slim.l2_regularizer(Config.model.reg_scale))
            dep_embedding = tf.get_variable('dep_embedding', [depvec.shape[0], depvec.shape[1]],
                                            initializer=tf.constant_initializer(depvec, tf.float32),
                                            regularizer=slim.l2_regularizer(Config.model.reg_scale))
            word = tf.nn.embedding_lookup(word_embedding, word)
            pos = tf.nn.embedding_lookup(pos_embedding, pos)
            dep = tf.nn.embedding_lookup(dep_embedding, dep)

            word = tf.reshape(word, [-1, word.shape[1] * word.shape[2]])
            pos = tf.reshape(pos, [-1, pos.shape[1] * pos.shape[2]])
            dep = tf.reshape(dep, [-1, dep.shape[1] * dep.shape[2]])

            return [word, pos, dep]

    def cube_activation(self, inputs):
        word, pos, dep = inputs
        with tf.variable_scope('fc_layer1'):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=None,
                                weights_regularizer=slim.l2_regularizer(Config.model.reg_scale)):
                fc1_1 = slim.fully_connected(word, Config.model.fc1_unit, biases_initializer=None)
                fc1_2 = slim.fully_connected(pos, Config.model.fc1_unit, biases_initializer=None)
                fc1_3 = slim.fully_connected(dep, Config.model.fc1_unit)
                fc1 = tf.pow(tf.add_n([fc1_1, fc1_2, fc1_3]), 3)
            outputs = slim.dropout(fc1, Config.model.dropout_keep_prob, is_training=self.is_training)
            return outputs

    def fc_layer(self, inputs):
        with tf.variable_scope('fc_layer2'):
            logits = slim.fully_connected(inputs, Config.data.num_dep * 2 + 1, activation_fn=None,
                                          weights_regularizer=slim.l2_regularizer(Config.model.reg_scale))
            return logits
