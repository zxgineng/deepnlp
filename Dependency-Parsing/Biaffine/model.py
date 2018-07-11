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
        self.loss, self.train_op, self.predictions, self.metrics = None, None, None, None
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
        arc_logits, label_logits = graph.build(self.inputs)

        arc_pred = tf.argmax(arc_logits, -1)
        arc_mask = tf.one_hot(arc_pred, tf.cast(tf.reduce_max(self.inputs['length']), tf.int32), True, False,
                              dtype=tf.bool)

        label_pred = tf.argmax(tf.boolean_mask(label_logits, arc_mask), -1)
        label_pred = tf.reshape(label_pred, tf.stack([-1, tf.reduce_max(self.inputs['length']) - 1]))

        self.predictions = {'arc': arc_pred, 'dep_id': label_pred}

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(arc_logits, label_logits)
            self._build_train_op()
            self._build_metric()

    def _build_loss(self, arc_logits, label_logits):
        weights = tf.sequence_mask(self.inputs['length'] - 1, dtype=tf.float32)  # remove root
        arc_loss = tf.losses.sparse_softmax_cross_entropy(self.targets['arc'], arc_logits, weights)

        arc_mask = tf.one_hot(self.targets['arc'], tf.cast(tf.reduce_max(self.inputs['length']), tf.int32), True, False,
                              dtype=tf.bool)  # [batch,length-1] -> [batch,length-1,length]
        label_logits = tf.boolean_mask(label_logits,
                                       arc_mask)  # [batch,length-1,length,channel] -> [batch*(length-1),channel]
        label_logits = tf.reshape(label_logits, tf.stack([-1, tf.reduce_max(self.inputs['length']) - 1,
                                                          Config.model.dep_num]))  # [batch*(length-1),channel] -> [batch,length-1,channel]
        label_loss = tf.losses.sparse_softmax_cross_entropy(self.targets['dep_id'], label_logits, weights)

        self.loss = arc_loss + label_loss

    def _build_train_op(self):
        global_step = tf.train.get_global_step()
        learning_rate = Config.train.initial_lr * tf.pow(0.75, tf.cast(global_step // 1500, tf.float32))

        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.9),
            learning_rate=learning_rate,
            clip_gradients=Config.train.max_gradient_norm)

    def _build_metric(self):
        weights = tf.sequence_mask(self.inputs['length'] - 1, dtype=tf.float32)
        self.metrics = {
            "UAS": tf.metrics.accuracy(self.targets['arc'], self.predictions['arc'], weights),
            'LAS': tf.metrics.accuracy(self.targets['arc'] * 1000 + self.targets['dep_id'],
                                       self.predictions['arc'] * 1000 + self.predictions['dep_id'], weights)
        }