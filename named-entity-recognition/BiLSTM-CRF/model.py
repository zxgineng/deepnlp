import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config
from network import Graph


class Model:
    def __init__(self):
        pass

    def model_fn(self, mode, features, labels):
        self.mode = mode
        self.features = features
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
        graph = Graph(self.mode)
        logits = graph.build(self.features)

        transition_params = tf.get_variable("transitions", [Config.model.fc_unit, Config.model.fc_unit])
        viterbi_sequence, _ = tf.contrib.crf.crf_decode(logits, transition_params, self.features['length'])

        self.predictions = viterbi_sequence
        tf.identity(self.predictions, 'prediction')

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(logits, transition_params)
            if self.mode == tf.estimator.ModeKeys.TRAIN:
                self._build_train_op()
            else:
                precision = tf.placeholder(tf.float32, None, 'p_ph')
                recall = tf.placeholder(tf.float32, None, 'r_ph')
                f1_measure = tf.placeholder(tf.float32, None, 'f1_ph')
                tf.summary.scalar('precision', precision, ['prf'], 'score')
                tf.summary.scalar('recall', recall, ['prf'], 'score')
                tf.summary.scalar('f1_measure', f1_measure, ['prf'], 'score')

    def _build_loss(self, logits, transition_params):
        with tf.variable_scope('loss'):
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, self.targets,
                                                                  self.features['length'], transition_params)
            self.loss = tf.reduce_mean(-log_likelihood)

    def _build_train_op(self):
        global_step = tf.train.get_global_step()

        learning_rate = Config.train.initial_lr

        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=tf.train.GradientDescentOptimizer(learning_rate),
            learning_rate=learning_rate,
            clip_gradients=Config.train.max_gradient_norm)
