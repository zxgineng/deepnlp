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
        graph = Graph(self.mode)
        outputs = graph.build(self.inputs)

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(outputs)
            self._build_train_op()

    def _build_loss(self, inputs):
        for_outputs = inputs[0]
        back_outputs = inputs[1]
        for_targets = tf.reshape(self.targets['for_labels'], [-1, 1])
        back_targets = tf.reshape(self.targets['back_labels'], [-1, 1])
        softmax_w = tf.get_variable('w', [Config.model.vocab_num, Config.model.embedding_size], tf.float32,
                                    slim.xavier_initializer())
        softmax_b = tf.get_variable('b', [Config.model.vocab_num], tf.float32, tf.constant_initializer(0.0))
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            for_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_w, softmax_b, for_targets, for_outputs,
                                                                 Config.train.sampled_num, Config.model.vocab_num))
            tf.summary.scalar('for-loss', for_loss)
            back_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_w, softmax_b, back_targets, back_outputs,
                                                                  Config.train.sampled_num, Config.model.vocab_num))
            tf.summary.scalar('back-loss', back_loss)
            self.loss = 0.5 * (for_loss + back_loss)

    def _build_train_op(self):
        global_step = tf.train.get_global_step()
        # learning_rate = Config.train.initial_lr / (Config.train.decay_rate * Config.train.epoch + 1)
        learning_rate = Config.train.initial_lr

        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=tf.train.AdagradOptimizer(learning_rate, 1.0),
            learning_rate=learning_rate,
            clip_gradients=Config.train.max_gradient_norm)
