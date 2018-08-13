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
        self.loss, self.train_op, self.predictions, self.evaluation_hooks, self.metrics = None, None, None, None, None
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
        graph = Graph()
        sen_vec = graph(self.inputs, self.mode)



        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self._build_loss(sen_vec)
            self._build_train_op()
            self._build_metric()

    def _build_loss(self, sen_matrix):
        #  rearange sen_matrix according to labels
        #  sort label and sen_matrix
        sorted_idx = tf.nn.top_k(self.targets, tf.shape(self.targets)[0]).indices
        sorted_label = tf.gather(self.targets, sorted_idx)
        sorted_sen = tf.gather(sen_matrix, sorted_idx)
        #  count num of each class
        class_count = tf.bincount(tf.cast(self.targets, tf.int32), dtype=tf.int64)[::-1]
        max_count = tf.reduce_max(class_count)
        #  non-zero
        non_zero_class_count = tf.boolean_mask(class_count, tf.cast(class_count, tf.bool))
        # build index matrix
        y_idx, x_idx = tf.meshgrid(tf.range(max_count), tf.range(tf.shape(non_zero_class_count, out_type=tf.int64)[0]))
        grid = tf.stack([x_idx, y_idx], -1)
        mask = tf.sequence_mask(non_zero_class_count)
        sparse_idx = tf.boolean_mask(grid, mask)  # valid idx
        # use sparse tensor to reshape sen_matrix
        shape = tf.stack([tf.shape(non_zero_class_count, out_type=tf.int64)[0], max_count])
        idx_matrix = tf.sparse_tensor_to_dense(
            tf.SparseTensor(sparse_idx, tf.range(tf.shape(self.targets, out_type=tf.int64)[0]), shape))
        labels = tf.unique(sorted_label).y
        # new sen_matrix [non_zero_class_num, max_count, dim]
        sen_matrix = tf.gather_nd(sorted_sen, tf.expand_dims(idx_matrix, -1))
        # cal sentence-attention
        relation_repre = tf.keras.layers.Embedding(12, tf.shape(sen_matrix, out_type=tf.int64)[-1])(labels)
        r = tf.expand_dims(relation_repre, -1)  # [non_zero_class_num,dim,1]
        score = tf.squeeze(tf.matmul(sen_matrix, r), -1)  # [non_zero_class_num,max_count]
        log_score = tf.exp(score - tf.reduce_max(score, -1, True)) * tf.cast(mask, tf.float32)
        softmax = log_score / tf.reduce_sum(log_score, -1, True)
        alpha = tf.expand_dims(softmax, -1)  # [non_zero_class_num,max_count,1]
        sentence_att = tf.reduce_sum(alpha * sen_matrix, 1)  # [non_zero_class_num, dim]
        logits = tf.keras.layers.Dense(Config.model.class_num)(sentence_att)

        xentropy = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)



    def _build_train_op(self):
        global_step = tf.train.get_global_step()
        if Config.train.switch_optimizer == 0:
            print('using Adadelta')
            optimizer = tf.train.AdadeltaOptimizer(Config.train.initial_lr)
            lr = Config.train.initial_lr
        else:
            print('using SGD')
            optimizer = tf.train.GradientDescentOptimizer(Config.train.sgd_lr)
            lr = Config.train.sgd_lr

        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=optimizer,
            learning_rate=lr)

    def _build_metric(self):
        self.metrics = {
            "accuracy": tf.metrics.accuracy(self.targets, self.predictions)
        }
