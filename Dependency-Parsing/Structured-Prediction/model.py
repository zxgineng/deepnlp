import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config
from network import Graph
from hooks import BeamSearchHook


class Model:
    def __init__(self):
        pass

    def model_fn(self, mode, features, labels):
        self.mode = mode
        self.inputs = features
        self.targets = labels
        self.loss, self.train_op, self.predictions, self.training_hooks = None, None, None, None
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            predictions=self.predictions,
            training_hooks=self.training_hooks)

    def build_graph(self):
        graph = Graph(self.mode)
        inputs_ph = {'word_id': tf.placeholder(tf.int64, [None, Config.model.word_feature_num], 'word_id'),
                     'pos_id': tf.placeholder(tf.int64, [None, Config.model.pos_feature_num], 'pos_id'),
                     'dep_id': tf.placeholder(tf.int64, [None, Config.model.dep_feature_num], 'dep_id')}
        logits = graph.build(inputs_ph)
        tf.identity(logits, 'scores')
        self.training_hooks = [BeamSearchHook(self.inputs, self.targets)]

        seg_id = tf.placeholder(tf.int64, [None], 'seg_id')
        beam_search_word = tf.placeholder(tf.int64, [Config.model.beam_size, None, Config.model.word_feature_num],
                                          'beam_search_word')
        beam_search_pos = tf.placeholder(tf.int64, [Config.model.beam_size, None, Config.model.pos_feature_num],
                                         'beam_search_pos')
        beam_search_dep = tf.placeholder(tf.int64, [Config.model.beam_size, None, Config.model.pos_feature_num],
                                         'beam_search_dep')
        beam_search_label = tf.placeholder(tf.int64,[Config.model.beam_size,None])

        action_num = Config.model.dep_num * 2 + 1

        all_scores = [[]]

        def condition(i, seg_id, beam_search_word, beam_search_pos, beam_search_dep,beam_search_label,all_scores):
            return tf.less(seg_id.shape[0] - 1, i)

        def body(i, seg_id, beam_search_word, beam_search_pos, beam_search_dep,beam_search_label,all_scores):
            inputs = {'word_id': tf.reshape(beam_search_word[:, seg_id[i]:seg_id[i + 1], :],
                                            [-1, Config.model.word_feature_num]),
                      'pos_id': tf.reshape(beam_search_pos[:, seg_id[i]:seg_id[i + 1], :],
                                           [-1, Config.model.pos_feature_num]),
                      'dep_id': tf.reshape(beam_search_dep[:, seg_id[i]:seg_id[i + 1], :],
                                           [-1, Config.model.dep_feature_num])}
            logits = graph.build(inputs)
            logits = tf.reshape(logits,[Config.model.beam_size,seg_id[i + 1]-seg_id[i],action_num])
            beam_search_label = beam_search_label[:,seg_id[i]:seg_id[i + 1]]
            one_sample_socres = tf.reduce_sum(tf.multiply(tf.one_hot(beam_search_label,seg_id[i + 1]-seg_id[i]),logits),[-1,-2])
            all_scores = tf.concat([all_scores,one_sample_socres],-1)

            return [i + 1, seg_id, beam_search_word, beam_search_pos, beam_search_dep,beam_search_label,all_scores]

        i = 0
        [i, seg_id, beam_search_word, beam_search_pos, beam_search_dep, beam_search_label, all_scores] =\
            tf.while_loop(condition, body, [i, seg_id, beam_search_word, beam_search_pos, beam_search_dep,beam_search_label,all_scores])

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(all_scores)
            self._build_train_op()

    def _build_loss(self, logits):
        labels = tf.zeros_like(logits,tf.int64)[:,0]
        tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
        self.loss = tf.losses.get_total_loss()

    def _build_train_op(self):
        global_step = tf.train.get_global_step()

        if Config.train.epoch <= 10:
            learning_rate = Config.train.initial_lr
        else:
            learning_rate = Config.train.initial_lr * 0.1

        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=tf.train.MomentumOptimizer(learning_rate, 0.9),
            learning_rate=learning_rate,
            clip_gradients=Config.train.max_gradient_norm)
