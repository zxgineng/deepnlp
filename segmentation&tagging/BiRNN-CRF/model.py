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
        graph = architecture.Graph(self.mode)
        logits = graph.build(self.features)

        transition_params = tf.get_variable("transitions", [Config.model.fc_unit,Config.model.fc_unit])


        viterbi_sequence, _ = tf.contrib.crf.crf_decode(logits, transition_params,self.features['length'])

        self.predictions = {'predictions': viterbi_sequence}

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(logits, transition_params)
            self._build_optimizer()

    def _build_loss(self,logits, transition_params):
        with tf.variable_scope('loss'):
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, self.targets['tagid'],
                                                                  self.features['length'], transition_params)
            self.loss = tf.reduce_mean(-log_likelihood)

    def _build_optimizer(self):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(Config.train.learning_rate, global_step,
                                                   Config.train.learning_decay_steps, Config.train.learning_decay_rate)
        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer='Adam',
            learning_rate=learning_rate,
            clip_gradients=Config.train.max_gradient_norm,
            summaries=['loss'],
            name="train_op")
