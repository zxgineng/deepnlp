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
        self.loss, self.train_op, self.predictions,self.metrics = None, None, None,None
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


        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(arc_logits, label_logits)
            self._build_train_op()
            self._build_metric()

    def _build_loss(self, arc_logits, label_logits):
        mask = tf.sequence_mask(self.inputs['length'],dtype=tf.float32)





    def _build_train_op(self):
        global_step = tf.train.get_global_step()

        if Config.train.epoch <= 10:
            learning_rate = Config.train.initial_lr
        else:
            learning_rate = Config.train.initial_lr * 0.1

        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=tf.train.MomentumOptimizer(learning_rate,0.9),
            learning_rate=learning_rate,
            clip_gradients=Config.train.max_gradient_norm)