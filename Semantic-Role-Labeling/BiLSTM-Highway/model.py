import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import os

from utils import Config
from network import Graph
from hooks import PRFScoreHook


class Model:
    def __init__(self):
        pass

    def model_fn(self, mode, features, labels):
        self.mode = mode
        self.inputs = features
        self.targets = labels
        self.loss, self.train_op, self.predictions, self.evaluation_hooks = None, None, None, None
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            predictions=self.predictions,
            evaluation_hooks=self.evaluation_hooks)

    def build_graph(self):
        graph = Graph()
        logits = graph(self.inputs, self.mode)

        def hard_constraints():
            ninf = -np.inf
            params = np.zeros([Config.model.class_num, Config.model.class_num])
            with open(os.path.join(Config.data.processed_path, Config.data.label_file)) as f:
                labels = f.read().splitlines()
                i = 0
                while i < len(labels):
                    j = 0
                    while j < len(labels):
                        if labels[i][0] == 'B' and labels[j][0] in ['B', 'S', 'O', 'r']:
                            params[i, j] = ninf
                        elif labels[i][0] == 'I' and labels[j][0] in ['B', 'S', 'O', 'r']:
                            params[i, j] = ninf
                        elif labels[i][0] == 'E' and labels[j][0] in ['I', 'E']:
                            params[i, j] = ninf
                        elif labels[i][0] == 'S' and labels[j][0] in ['I', 'E']:
                            params[i, j] = ninf
                        elif labels[i][0] == 'O' and labels[j][0] in ['I', 'E']:
                            params[i, j] = ninf
                        elif labels[i][0] == 'r' and labels[j][0] in ['I', 'E', 'r']:
                            params[i, j] = ninf
                        elif labels[i][0] == 'B' and labels[j][0] in ['I', 'E'] and labels[i][1:] != labels[j][1:]:
                            params[i, j] = ninf
                        elif labels[i][0] == 'I' and labels[j][0] in ['I', 'E'] and labels[i][1:] != labels[j][1:]:
                            params[i, j] = ninf
                        j += 1
                    i += 1
            return params

        transition_params = hard_constraints()
        transition_params = tf.constant(transition_params, tf.float32)

        viterbi_sequence, _ = tf.contrib.crf.crf_decode(logits, transition_params,
                                                        tf.cast(self.inputs['length'], tf.int32))

        self.predictions = viterbi_sequence
        tf.identity(self.predictions, 'prediction')

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(logits, transition_params)
            if self.mode == tf.estimator.ModeKeys.TRAIN:
                self._build_train_op()
            else:
                self.evaluation_hooks = [PRFScoreHook()]
                precision = tf.placeholder(tf.float32, None, 'p_ph')
                recall = tf.placeholder(tf.float32, None, 'r_ph')
                f1_measure = tf.placeholder(tf.float32, None, 'f1_ph')
                tf.summary.scalar('precision', precision, ['prf'], 'score')
                tf.summary.scalar('recall', recall, ['prf'], 'score')
                tf.summary.scalar('f1_measure', f1_measure, ['prf'], 'score')

    def _build_loss(self, logits, transition_params):

        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, self.targets,
                                                              self.inputs['length'], transition_params)
        self.loss = tf.reduce_mean(-log_likelihood)

    def _build_train_op(self):

        def clip_gradient(grads_and_vars):
            clipped = [(tf.clip_by_norm(grad, Config.train.clip_gradients), var) for grad, var in grads_and_vars]
            return clipped

        global_step = tf.train.get_global_step()
        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=tf.train.AdadeltaOptimizer(1.0, epsilon=1e-6),
            learning_rate=1.0,
            clip_gradients=clip_gradient)
