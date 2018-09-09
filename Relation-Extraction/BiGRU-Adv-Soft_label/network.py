import tensorflow as tf

from utils import Config
from data_loader import load_pretrained_vec


class Graph(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Graph, self).__init__(**kwargs)

        self.embedding_layer = Embedding_Layer(name='embedding_layer')
        self.encoding_layer = Encoding_Layer(name='encoding_layer')
        self.selective_attention = Selective_Attention(name='selective_attention')
        self.softmax_dense = tf.keras.layers.Dense(Config.model.class_num)

    def call(self, inputs, targets, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            training = True
        else:
            training = False

        word_id = inputs['word_id']
        en_indicator = inputs['en_indicator']
        length = inputs['length']

        embedded = self.embedding_layer(word_id, en_indicator, training)
        sen_matrix = self.encoding_layer(embedded, length)  # [B,2*dim]

        if training:
            sen_matrix, labels = self.selective_attention(sen_matrix, targets)
            sen_matrix = tf.nn.dropout(sen_matrix, Config.model.dropout_keep_prob)

            # adversarial training
            if Config.train.adversarial_training:
                logits = self.softmax_dense(sen_matrix)
                loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
                g = tf.gradients(loss, self.embedding_layer.origin_embedded)
                e = Config.model.epsilon * tf.stop_gradient(tf.nn.l2_normalize(g))[0]
                embedded = tf.pad(e, [[0, 0], [0, 0], [0, Config.model.entity_indicator_size]]) + embedded
                sen_matrix = self.encoding_layer(embedded, length)
                sen_matrix, labels = self.selective_attention(sen_matrix, targets)
                sen_matrix = tf.nn.dropout(sen_matrix, Config.model.dropout_keep_prob)
        else:
            labels = None

        logits = self.softmax_dense(sen_matrix)

        return logits, labels


class Embedding_Layer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Embedding_Layer, self).__init__(**kwargs)

        wordvec = load_pretrained_vec()
        self.word_embedding = tf.keras.layers.Embedding(wordvec.shape[0], wordvec.shape[1],
                                                        tf.constant_initializer(wordvec, tf.float32),
                                                        name='word_embedding')

    def call(self, word_id, en_indicator, training):
        word_embedded = self.word_embedding(word_id)
        if training:
            word_embedded = tf.nn.dropout(word_embedded, Config.model.dropout_keep_prob)
        en_indicator = tf.cast(tf.tile(tf.expand_dims(en_indicator, -1), [1, 1, Config.model.entity_indicator_size]),
                               tf.float32)
        outputs = tf.concat([word_embedded, en_indicator], -1)
        self.origin_embedded = word_embedded
        return outputs


class Encoding_Layer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Encoding_Layer, self).__init__(**kwargs)
        self.fw_gru = tf.nn.rnn_cell.GRUCell(Config.model.gru_units, kernel_initializer=tf.orthogonal_initializer())
        self.bw_gru = tf.nn.rnn_cell.GRUCell(Config.model.gru_units, kernel_initializer=tf.orthogonal_initializer())
        self.softmax_dense = tf.keras.layers.Dense(1)

    def call(self, inputs, length):
        outputs, states = tf.nn.bidirectional_dynamic_rnn(self.fw_gru, self.bw_gru, inputs, length, dtype=tf.float32)
        gru_outputs = tf.concat(outputs, -1)  # [B,seq,2*dim]
        outputs = gru_outputs[:,-1,:]  # [B,2*dim]
        return outputs


class Selective_Attention(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Selective_Attention, self).__init__(**kwargs)

    def call(self, inputs, targets):
        labels = targets['label']
        entity_pair_id = targets['entity_pair_id']
        #  sort label and inputs
        entity_pair_id, sorted_idx = tf.nn.top_k(entity_pair_id, tf.shape(entity_pair_id)[0])
        labels = tf.gather(labels, sorted_idx)
        sen_matrix = tf.gather(inputs, sorted_idx)
        #  count num of each pair
        pair_count = tf.bincount(tf.cast(entity_pair_id, tf.int32), dtype=tf.int64)[::-1]
        max_pair_count = tf.reduce_max(pair_count)
        #  non-zero
        pair_count = tf.boolean_mask(pair_count, tf.cast(pair_count, tf.bool))
        labels = tf.gather(labels, tf.cumsum(pair_count) - 1)
        # build index matrix
        shape = tf.stack([tf.shape(pair_count, out_type=tf.int64)[0], max_pair_count])
        temp = tf.zeros(shape, tf.int64)
        mask = tf.sequence_mask(pair_count)
        temp = temp + tf.cast(mask, tf.int64)
        sparse_idx = tf.where(tf.equal(temp, 1))  # valid idx
        # use sparse tensor to reshape sen_matrix
        idx_matrix = tf.sparse_tensor_to_dense(
            tf.SparseTensor(sparse_idx, tf.range(tf.shape(entity_pair_id, out_type=tf.int64)[0]), shape))
        # new sen_matrix [pair_num, max_pair_count, 2*dim]
        sen_matrix = tf.gather_nd(sen_matrix, tf.expand_dims(idx_matrix, -1))
        # cal sentence-attention
        relation_repre = tf.keras.layers.Embedding(Config.model.class_num, 2 * Config.model.gru_units)(labels)
        r = tf.expand_dims(relation_repre, -1)  # [pair_num,dim,1]
        score = tf.squeeze(tf.matmul(sen_matrix, r), -1)  # [pair_num,max_pair_count]
        log_score = tf.exp(score - tf.reduce_max(score, -1, True)) * tf.cast(mask, tf.float32)
        softmax = log_score / (tf.reduce_sum(log_score, -1, True) + 1e-8)
        alpha = tf.expand_dims(softmax, -1)  # [pair_num,max_pair_count,1]
        sen_att = tf.reduce_sum(alpha * sen_matrix, 1)  # [pair_num, 2*dim]
        return sen_att, labels
