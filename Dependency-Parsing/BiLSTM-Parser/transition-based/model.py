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
            predictions=self.predictions)

    def build_graph(self):
        graph = architecture.Graph(self.mode)
        lstm_ouputs,mlp_func = graph.build(self.features)
        lstm_ouputs = tf.squeeze(lstm_ouputs,0)




        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(lstm_ouputs,mlp_func)
            self._build_optimizer()

    def _build_loss(self,lstm_ouputs,mlp_func):
        length = self.features['length'][0]
        stack = lstm_ouputs[:3]
        buffer = lstm_ouputs[3:]


        def condition(stack,buffer):
            return buffer.shape[0] == 1 and stack.shape[0] == 0

        def body(stack,buffer):
            inputs = tf.concat([stack[-3:],buffer[0]],0) # 待定
            scores =





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
