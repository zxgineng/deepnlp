import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config
import architecture


class Model:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.mode = mode
        self.params = params
        self.features = features
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
            predictions={'predictions': self.predictions})

    def build_graph(self):
        graph = architecture.Graph(self.mode)
        logits = graph.build(self.features)

        self.predictions = logits

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(logits)
            self._build_optimizer()
            self._build_metric(logits)

    def _build_loss(self, logits):
        with tf.variable_scope('loss'):
            tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self.targets)
            self.loss = tf.losses.get_total_loss()

    def _build_optimizer(self):
        global_step = tf.train.get_global_step()
        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=Config.train.optimizer,
            learning_rate=Config.train.learning_rate,
            name="train_op")

    def _build_metric(self, logits):
        self.metrics = {
            'accuracy': tf.metrics.accuracy(self.targets, tf.argmax(logits, -1))
        }
