import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config
from network import Graph
from hooks import EvalHook


class Model:
    def __init__(self):
        pass

    def model_fn(self, mode, features, labels):
        self.mode = mode
        self.inputs = features
        self.targets = labels
        self.loss, self.train_op, self.predictions,self.evaluation_hooks = None, None, None, None
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            predictions=self.predictions,
            evaluation_hooks = self.evaluation_hooks)

    def build_graph(self):
        graph = Graph()
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            logits = graph(self.inputs, self.mode)
            self._build_loss(logits)
            self._build_train_op()
        else:
            inputs = {'tree_word_id': tf.placeholder(tf.int64, [None, None], name='tree_word_id'),
                      'tree_pos_id': tf.placeholder(tf.int64, [None, None], name='tree_pos_id'),
                      'buff_word_id': tf.placeholder(tf.int64, [None, None], name='buff_word_id'),
                      'buff_pos_id': tf.placeholder(tf.int64, [None, None], name='buff_pos_id'),
                      'history_action_id': tf.placeholder(tf.int64, [None, None], name='history_action_id'),
                      'comp_head_order': tf.placeholder(tf.int64, [None, None], name='comp_head_order'),
                      'comp_dep_order': tf.placeholder(tf.int64, [None, None], name='comp_dep_order'),
                      'comp_rel_id': tf.placeholder(tf.int64, [None, None], name='comp_rel_id'),
                      'is_leaf': tf.placeholder(tf.int64, [None, None], name='is_leaf'),
                      'stack_order': tf.placeholder(tf.int64, [None, None], name='stack_order'),
                      'stack_length': tf.placeholder(tf.int64, [None], name='stack_length'),
                      'buff_length': tf.placeholder(tf.int64, [None], name='buff_length'),
                      'history_action_length': tf.placeholder(tf.int64, [None], name='history_action_length')}

            logits = graph(inputs, self.mode)
            prob = tf.nn.softmax(logits)
            self.loss = tf.constant(0)  # the parsing doesn't follow expected transition when eval, loss makes no sense
            head_acc = tf.placeholder(tf.float32, None, 'head_ph')
            dep_acc = tf.placeholder(tf.float32, None, 'dep_ph')
            tf.summary.scalar('UAS', head_acc, ['acc'], 'score')
            tf.summary.scalar('LAS', dep_acc, ['acc'], 'score')

            self.evaluation_hooks = [EvalHook()]
            prediction = tf.argmax(logits, -1)
            self.predictions = prediction

    def _build_loss(self, logits):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=self.targets['transition'], logits=logits)
        reg_loss = Config.train.reg_scale * tf.reduce_sum(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'embedding' not in v.name and 'bias' not in v.name])
        self.loss = loss + reg_loss

    def _build_train_op(self):
        global_step = tf.train.get_global_step()
        learning_rate = Config.train.initial_lr / (1 + Config.train.lr_decay * Config.train.epoch)

        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=tf.train.MomentumOptimizer(learning_rate, 0.9),
            learning_rate=learning_rate,
            clip_gradients=Config.train.max_gradient_norm)
