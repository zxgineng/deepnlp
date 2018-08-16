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
        self.loss, self.train_op, self.predictions = None, None, None
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            predictions=self.predictions)

    def build_graph(self):
        graph = Graph()
        logits, labels = graph(self.inputs, self.targets, self.mode)

        self.predictions = tf.argmax(logits, -1)
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(logits, labels)
            self._build_train_op()

    def _build_loss(self, logits, labels):
        if self.mode == tf.estimator.ModeKeys.EVAL:
            labels = self.targets
        elif Config.train.soft_label:

            def soft_label():
                confidence = tf.ones([Config.model.class_num], tf.float32) * 0.7
                soft_score = logits + confidence * tf.reduce_max(logits, -1, True) \
                             * tf.one_hot(labels, Config.model.class_num, dtype=tf.float32)
                soft_labels = tf.argmax(soft_score, -1)
                return soft_labels

            global_step = tf.train.get_global_step()
            labels = tf.cond(tf.greater(global_step, Config.train.soft_label_start),
                             soft_label, lambda: labels)

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    def _build_train_op(self):
        global_step = tf.train.get_global_step()
        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=tf.train.AdamOptimizer(Config.train.initial_lr),
            learning_rate=Config.train.initial_lr)
