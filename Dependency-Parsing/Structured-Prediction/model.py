import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config
from network import Graph
from hooks import BeamSearchHook

class Model:
    def __init__(self):
        pass

    def model_fn(self, mode, features, labels):
        self.mode = mode
        self.inputs = features
        self.targets = labels
        self.loss, self.train_op, self.predictions,self.training_hooks = None, None, None,None
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            predictions=self.predictions,
            training_hooks=self.training_hooks)

    def build_graph(self):
        graph = Graph(self.mode)
        inputs_ph = {'word_id':tf.placeholder(tf.int64,[None,Config.model.word_token_num],'word_id'),
                     'pos_id':tf.placeholder(tf.int64,[None,Config.model.pos_token_num],'pos_id'),
                     'dep_id':tf.placeholder(tf.int64,[None,Config.model.dep_token_num],'dep_id')}
        logits = graph.build(inputs_ph)
        tf.identity(logits,'scores')
        self.training_hooks = [BeamSearchHook(self.inputs,self.targets)]

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(logits)
            self._build_train_op()

    def _build_loss(self,logits):
        length = 2 * self.targets['length'] - 1
        logits = tf.reshape(logits,[-1,Config.model.beam_size,tf.reduce_max(length)])
        # todo 添加 early-stop mask
        logits = tf.reduce_sum(logits,-1)
        labels = tf.zeros_like(logits)[:,0]
        tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
        self.loss = tf.losses.get_total_loss()



    def _build_train_op(self):
        # global_step = tf.train.get_global_step()
        #
        # if Config.train.epoch <= 10:
        #     learning_rate = Config.train.initial_lr
        # else:
        #     learning_rate = Config.train.initial_lr * 0.1
        #
        # self.train_op = slim.optimize_loss(
        #     self.loss, global_step,
        #     optimizer=tf.train.MomentumOptimizer(learning_rate,0.9),
        #     learning_rate=learning_rate,
        #     clip_gradients=Config.train.max_gradient_norm)
