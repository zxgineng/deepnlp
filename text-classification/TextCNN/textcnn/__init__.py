from utils import Config
import tensorflow as tf
from tensorflow.contrib import slim


class Graph:
    def __init__(self, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = True
        else:
            self.is_training = False

    def build(self, input_data):

        embedding_input = self.build_embed(input_data)
        conv_output = self.build_conv_layers(embedding_input)
        logits = self.build_fully_connected_layers(conv_output)
        prediction = tf.argmax(logits, 1)
        return logits, prediction

    def build_embed(self, input_data):
        with tf.variable_scope("Embedding"):
            embedding = tf.get_variable("embedding", [Config.data.vocab_size, Config.model.embedding_dim], tf.float32)
            embed = tf.nn.embedding_lookup(embedding, input_data)
            return embed

    def build_conv_layers(self, embedding_input):
        with tf.variable_scope("Convolutions"):
            pooled_outputs = self._build_conv_maxpool(embedding_input)
            concat_pooled = tf.concat(pooled_outputs, 1)
            dropout = slim.dropout(concat_pooled, Config.model.dropout_keep_prob, is_training=self.is_training)

            return dropout

    def _build_conv_maxpool(self, embedding_input):
        pooled_outputs = []
        for kernel_size in Config.model.kernel_sizes:
            with tf.variable_scope(f"conv-maxpool-{kernel_size}-filter"):
                conv = tf.layers.conv1d(embedding_input, Config.model.num_filters, kernel_size,
                                        activation=tf.nn.relu, kernel_initializer=slim.xavier_initializer())

                pool = tf.reduce_max(conv, 1)
                pooled_outputs.append(pool)
        return pooled_outputs

    def build_fully_connected_layers(self, conv_output):
        with tf.variable_scope("fully-connected"):
            return slim.fully_connected(conv_output, Config.data.num_classes, activation_fn=None)
