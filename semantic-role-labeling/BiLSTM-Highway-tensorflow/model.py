import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import os

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

        def bio_constraints():
            ninf = -np.inf
            params = np.zeros([Config.model.num_class, Config.model.num_class])
            with open(os.path.join(Config.data.base_path, 'tags.txt')) as f:
                tags = f.read().splitlines()
                i = 0
                while i < len(tags):
                    j = i + 1
                    while j < len(tags):
                        if tags[i][0] == 'B' and tags[j][0] == 'I' and tags[i][1:] != tags[j][1:]:
                            params[i, j] = ninf
                        j += 1
                    i += 1
            return params

        params = bio_constraints()

        viterbi_sequence, _ = tf.contrib.crf.crf_decode(logits, tf.constant(params, tf.float32),
                                                        self.features['length'])

        self.predictions = viterbi_sequence

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(logits)
            self._build_optimizer()
            self._build_metric(viterbi_sequence)

    def _build_loss(self, logits):
        with tf.variable_scope('loss'):
            mask = tf.sequence_mask(self.features['length'],dtype=tf.float32)
            tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self.targets,weights=mask)
            self.loss = tf.losses.get_total_loss()

    def _build_optimizer(self):
        global_step = tf.train.get_global_step()
        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=tf.train.AdadeltaOptimizer(Config.train.learning_rate,epsilon=1e-6),
            learning_rate=Config.train.learning_rate,
            clip_gradients= Config.train.clip_gradients,
            name="train_op")

    def _build_metric(self, viterbi_sequence):
        precision = tf.metrics.precision(self.targets, viterbi_sequence)
        recall = tf.metrics.recall(self.targets, viterbi_sequence)
        self.metrics = {
            'precision': precision,
            'recall': recall,
        }
