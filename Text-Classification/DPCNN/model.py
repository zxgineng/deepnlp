import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config
from network import Graph


class Model:
    def __init__(self):
        pass

    def model_fn(self, mode, features, labels):
        self.mode = mode
        self.inputs = features
        self.targets = labels
        self.loss, self.train_op, self.metrics, self.predictions = None, None, None, None
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=self.metrics,
            predictions=self.predictions)

    def build_graph(self):
        graph = Graph(self.mode)
        logits = graph.build(self.inputs)
        self.predictions = tf.argmax(logits, 1)

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(logits)
            self._build_train_op()
            self._build_metric()

    def _build_loss(self, logits):
        with tf.variable_scope('loss'):
            tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self.targets, scope='loss')
            self.loss = tf.losses.get_total_loss()

    def _build_train_op(self):
        global_step = tf.train.get_global_step()

        if Config.train.epoch < Config.train.max_epoch * 0.8:
            learning_rate = Config.train.learning_rate
        else:
            learning_rate = Config.train.learning_rate * 0.1

        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=tf.train.MomentumOptimizer(learning_rate, 0.9),
            learning_rate=learning_rate)

    def _build_metric(self):
        self.metrics = {
            "accuracy": tf.metrics.accuracy(self.targets, self.predictions)
        }
