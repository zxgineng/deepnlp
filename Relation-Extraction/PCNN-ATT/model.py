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
        self.loss, self.train_op, self.predictions= None, None, None
        self._build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            predictions=self.predictions)

    def _build_graph(self):
        graph = Graph()
        logits, labels = graph(self.inputs, self.targets, self.mode)

        self.predictions = tf.argmax(logits, -1)
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(logits, labels)
            self._build_train_op()

    def _build_loss(self, logits, labels):
        if self.mode == tf.estimator.ModeKeys.EVAL:
            labels = self.targets
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    def _build_train_op(self):
        global_step = tf.train.get_global_step()
        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=tf.train.MomentumOptimizer(Config.train.initial_lr, 0.9),
            learning_rate=Config.train.initial_lr)
