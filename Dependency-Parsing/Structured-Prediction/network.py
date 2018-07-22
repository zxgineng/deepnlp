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
        with tf.variable_scope('network',reuse=tf.AUTO_REUSE):
            word_id = inputs['word_feature_id']
            pos_id = inputs['pos_feature_id']
            dep_id = inputs['dep_feature_id']
            net = self._embedding(word_id, pos_id, dep_id)
            net = self._cube_activation(net)
            logits = self._fc_layer(net)
            return logits

    def _embedding(self, word_id, pos_id, dep_id):
        wordvec = load_pretrained_vec()
        embedding = tf.get_variable('word_embedding', [wordvec.shape[0], wordvec.shape[1]],
                                    initializer=tf.constant_initializer(wordvec, tf.float32))
        word_embedding = tf.nn.embedding_lookup(embedding, word_id)

        embedding = tf.get_variable('pos_embedding', [Config.model.pos_num, Config.model.pos_embedding_size])
        pos_embedding = tf.nn.embedding_lookup(embedding, pos_id)

        embedding = tf.get_variable('dep_embedding',[Config.model.dep_num, Config.model.dep_embedding_size])
        dep_embedding = tf.nn.embedding_lookup(embedding, dep_id)

        outputs = tf.concat([slim.flatten(word_embedding), slim.flatten(pos_embedding),slim.flatten(dep_embedding)], -1)
        return outputs


    def _cube_activation(self, inputs):
        net = slim.fully_connected(inputs,Config.model.fc_unit,None)
        net = tf.pow(net,3)
        outputs = slim.dropout(net,Config.model.dropout_keep_prob,is_training=self.is_training)
        return outputs

    def _fc_layer(self, inputs):
        logits = slim.fully_connected(inputs, (Config.model.dep_num - 1) * 2 + 1, activation_fn=None)
        return logits
