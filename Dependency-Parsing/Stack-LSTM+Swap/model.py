import tensorflow as tf

from utils import Config
from network import Graph


class Model:
    def __init__(self):
        pass

    def train(self,input_fn):
        self.mode = 'train'

        features,labels = input_fn()
        self._model_fn(features,labels)


    def _model_fn(self, features, labels):
        self.inputs = features
        self.targets = labels
        self.loss, self.train_op, self.predictions, self.training_hooks,self.evaluation_hooks = None, None, None, None,None
        self._build_graph()


    def _build_graph(self):
        graph = Graph(self.mode)
        logits = graph.build(self.inputs)
        print(logits.shape)
        exit()



    #
    #     if self.mode != tf.estimator.ModeKeys.PREDICT:
    #         self._build_loss(logits, transition_params)
    #         self._build_train_op()
    #
    #
    #
    # def _build_loss(self, logits, transition_params):
    #     with tf.variable_scope('loss'):
    #         log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, self.targets,
    #                                                               self.features['length'], transition_params)
    #         self.loss = tf.reduce_mean(-log_likelihood)
    #
    #
    # def _build_train_op(self):
    #     global_step = tf.train.get_global_step()
    #
    #     if Config.train.epoch <= 10:
    #         learning_rate = Config.train.initial_lr
    #     else:
    #         learning_rate = Config.train.initial_lr * 0.1
    #
    #     self.train_op = slim.optimize_loss(
    #         self.loss, global_step,
    #         optimizer=tf.train.MomentumOptimizer(learning_rate, 0.9),
    #         learning_rate=learning_rate,
    #         clip_gradients=Config.train.max_gradient_norm)
