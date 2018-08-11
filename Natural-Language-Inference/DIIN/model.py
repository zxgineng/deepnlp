import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config
from network import Graph
from hooks import SwitchOptimizerHook


class Model:
    def __init__(self):
        pass

    def model_fn(self, mode, features, labels):
        self.mode = mode
        self.inputs = features
        self.targets = labels
        self.loss, self.train_op, self.predictions, self.evaluation_hooks, self.metrics = None, None, None, None, None
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=self.metrics,
            predictions=self.predictions,
            evaluation_hooks=self.evaluation_hooks)

    def build_graph(self):
        graph = Graph()
        logits = graph(self.inputs, self.mode)

        self.predictions = tf.argmax(logits, -1)

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(logits)
            self._build_train_op()
            self.evaluation_hooks = [SwitchOptimizerHook()]

    def _build_loss(self, logits):

        xentropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.targets, logits=logits)

        l2_ratio = Config.train.l2_full_ratio * tf.nn.sigmoid(Config.train.epoch / 2.5 - 1)
        l2_loss = l2_ratio * tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                            if 'word_embedding' not in v.name and 'pos_embedding' not in v.name])

        l2_constraint = Config.train.constraint_scale * tf.reduce_sum(
            [tf.reduce_sum(tf.squared_difference(x, y)) for x, y in
             zip(tf.trainable_variables('graph/p_encoding_layer'),
                 tf.trainable_variables('graph/h_encoding_layer'))])

        self.loss = xentropy_loss + l2_loss + l2_constraint

    def _build_train_op(self):
        global_step = tf.train.get_global_step()
        if Config.train.switch_optimizer == 0:
            print('using Adadelta')
            optimizer = tf.train.AdadeltaOptimizer(Config.train.initial_lr)
            lr = Config.train.initial_lr
        else:
            print('using SGD')
            optimizer = tf.train.GradientDescentOptimizer(Config.train.sgd_lr)
            lr = Config.train.sgd_lr

        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=optimizer,
            learning_rate=lr)

    def _build_metric(self):
        self.metrics = {
            "accuracy": tf.metrics.accuracy(self.targets, self.predictions)
        }
