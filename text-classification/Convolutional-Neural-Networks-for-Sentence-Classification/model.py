import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config
import textcnn


class Model:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.mode = mode
        self.params = params
        self.inputs = features['inputs']
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
            predictions={"prediction": self.predictions})

    def build_graph(self):
        graph = textcnn.Graph(self.mode)
        output, predictions = graph.build(self.inputs)

        self.predictions = predictions
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(output)
            self._build_optimizer()
            self._build_metric()

    def _build_loss(self, logits):
        with tf.variable_scope('loss'):
            self.loss = tf.losses.sparse_softmax_cross_entropy(logits=logits,labels=self.targets,scope='loss')

    def _build_optimizer(self):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(Config.train.learning_rate,global_step,Config.train.learning_decay_steps,Config.train.learning_decay_rate)
        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer='Adam',
            learning_rate=learning_rate,
            summaries=['loss'],
            name="train_op")

    def _build_metric(self):
        self.metrics = {
            "accuracy": tf.metrics.accuracy(self.targets, self.predictions)
        }

