import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config
from network import Graph
from hooks import BeamTrainHook


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

        inputs = {'word_feature_id': tf.placeholder(tf.int64,[None,Config.model.word_feature_num],'word_feature_id'),
                  'pos_feature_id': tf.placeholder(tf.int64,[None,Config.model.pos_feature_num],'pos_feature_id'),
                  'dep_feature_id': tf.placeholder(tf.int64,[None,Config.model.dep_feature_num],'dep_feature_id')}
        logits = graph.build(inputs)
        tf.identity(logits, 'scores')

        slicer = tf.placeholder(tf.int64, [None], 'slicer')
        # save different bs length of each sentence
        bs_tran_length = tf.placeholder(tf.int64, [None, Config.model.beam_size],'bs_tran_length')

        beam_search_word = tf.placeholder(tf.int64, [Config.model.beam_size, None, Config.model.word_feature_num],
                                          'beam_search_word')
        beam_search_pos = tf.placeholder(tf.int64, [Config.model.beam_size, None, Config.model.pos_feature_num],
                                         'beam_search_pos')
        beam_search_dep = tf.placeholder(tf.int64, [Config.model.beam_size, None, Config.model.dep_feature_num],
                                         'beam_search_dep')
        beam_search_action = tf.placeholder(tf.int64, [Config.model.beam_size, None],'beam_search_action')

        action_num = (Config.model.dep_num - 1) * 2 + 2  # exclude NULL

        all_scores = tf.constant([],tf.float32)

        def condition(i, slicer,bs_tran_length, beam_search_word, beam_search_pos, beam_search_dep, beam_search_action, scores):
            return tf.less(i,tf.shape(bs_tran_length)[0])  # one loop for all beam search of one sentence

        def body(i, slicer,bs_tran_length, beam_search_word, beam_search_pos, beam_search_dep, beam_search_action, scores):

            inputs = {'word_feature_id': tf.reshape(beam_search_word[:, slicer[i]:slicer[i + 1], :],
                                                    [-1, Config.model.word_feature_num]),
                      'pos_feature_id': tf.reshape(beam_search_pos[:, slicer[i]:slicer[i + 1], :],
                                                   [-1, Config.model.pos_feature_num]),
                      'dep_feature_id': tf.reshape(beam_search_dep[:, slicer[i]:slicer[i + 1], :],
                                                   [-1, Config.model.dep_feature_num])}

            logits = graph.build(inputs)

            logits = tf.reshape(logits, [Config.model.beam_size, -1, action_num])

            action = beam_search_action[:, slicer[i]:slicer[i + 1]]

            bs_scores_seq = tf.reduce_sum(tf.multiply(tf.one_hot(action, action_num), logits),-1)  # [beam_size,max_len]
            mask = tf.sequence_mask(bs_tran_length[i],dtype=tf.float32)  # [beam_size,max_len]
            bs_scores_seq = mask * bs_scores_seq
            one_sample_scores = tf.reduce_sum(bs_scores_seq,-1)/tf.cast(bs_tran_length[i],tf.float32)  # [beam_size]

            scores = tf.concat([scores, one_sample_scores], -1)

            return [i + 1, slicer, bs_tran_length,beam_search_word, beam_search_pos, beam_search_dep, beam_search_action, scores]

        i = 0
        [i, slicer,bs_tran_length, beam_search_word, beam_search_pos, beam_search_dep, beam_search_action, scores] = \
            tf.while_loop(condition, body,
                          [i, slicer,bs_tran_length, beam_search_word, beam_search_pos, beam_search_dep, beam_search_action,all_scores],
                          [tf.TensorShape(None),slicer.get_shape(),bs_tran_length.get_shape(),beam_search_word.get_shape(),beam_search_pos.get_shape(),
                           beam_search_dep.get_shape(),beam_search_action.get_shape(),tf.TensorShape([None])])

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(scores)
            self._build_train_op()
            self.training_hooks = [BeamTrainHook(self.inputs, self.targets)]

    def _build_loss(self, logits):
        logits = tf.reshape(logits,[-1,Config.model.beam_size])
        labels = tf.zeros_like(logits, tf.int64)[:, 0]
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        reg_loss  = Config.train.reg_scale * tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.loss = loss + reg_loss

    def _build_train_op(self):
        global_step = tf.train.get_global_step()
        learning_rate = Config.train.initial_lr

        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=tf.train.AdagradOptimizer(learning_rate),
            learning_rate=learning_rate)
