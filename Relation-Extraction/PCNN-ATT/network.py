import tensorflow as tf

from utils import Config
from data_loader import load_pretrained_vec


class Graph(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Graph, self).__init__(**kwargs)

        self.embedding_layer = Embedding_Layer(name='embedding_layer')
        self.cnn_layer = CNN_Layer(name='cnn_layer')
        self.pool_layer = Pool_Layer(name='pool_layer')
        self.selective_attention = Selective_Attention(name='selective_attention')
        self.softmax_dense = tf.keras.layers.Dense(Config.model.class_num)

    def call(self, inputs, targets, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
        else:
            is_training = False

        word_id = inputs['word_id']
        pos_1 = inputs['pos_1']
        pos_2 = inputs['pos_2']
        en1_pos = inputs['en1_pos']
        en2_pos = inputs['en2_pos']
        embedded = self.embedding_layer(word_id, pos_1, pos_2)
        net = self.cnn_layer(embedded)
        sen_matrix = self.pool_layer(net, en1_pos, en2_pos)  # [B,3*dim]

        if is_training:
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
        self.pos_embedding = tf.keras.layers.Embedding(201, Config.model.position_embedding_size,
                                                       name='position_embedding')

    def call(self, word_id, pos_1, pos_2):
        word_embedded = self.word_embedding(word_id)
        pos_1_embedded = self.pos_embedding(pos_1)
        pos_2_embedded = self.pos_embedding(pos_2)
        outputs = tf.concat([word_embedded, pos_1_embedded, pos_2_embedded], -1)
        return outputs


class CNN_Layer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(CNN_Layer, self).__init__(**kwargs)

        self.conv1d = tf.keras.layers.Conv1D(Config.model.cnn_filters, 3, 1, 'SAME', activation=tf.nn.relu)

    def call(self, inputs):
        net = self.conv1d(inputs)
        return net


class Pool_Layer(tf.keras.Model):
    """piece pooling"""

    def __init__(self, **kwargs):
        super(Pool_Layer, self).__init__(**kwargs)

    def call(self, inputs, en1_pos, en2_pos):
        if Config.train.piece_pooling:
            seq_length, dim = tf.shape(inputs)[1], tf.shape(inputs)[2]

            mask1 = tf.expand_dims(tf.sequence_mask(en1_pos + 1, seq_length, tf.float32), -1)
            mask1 = tf.tile(mask1, [1, 1, dim])
            part = tf.where(tf.equal(mask1, 1.0), inputs, mask1)
            max1 = tf.reduce_max(part, 1)  # [B,dim]

            mask2 = tf.expand_dims(tf.sequence_mask(en2_pos + 1, seq_length, tf.float32), -1)
            mask2 = tf.tile(mask2, [1, 1, dim])
            part = tf.where(tf.equal(mask2 - mask1, 1.0), inputs, mask2 - mask1)
            max2 = tf.reduce_max(part, 1)  # [B,dim]

            mask3 = 1.0 - mask2
            part = tf.where(tf.equal(mask3, 1.0), inputs, mask3)
            max3 = tf.reduce_max(part, 1)  # [B,dim]
            outputs = tf.reshape(tf.stack([max1, max2, max3], -1), [-1, 3 * Config.model.cnn_filters])
        else:
            outputs = tf.reduce_max(inputs, 1)
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
        # new sen_matrix [pair_num, max_pair_count, dim]
        sen_matrix = tf.gather_nd(sen_matrix, tf.expand_dims(idx_matrix, -1))
        # cal sentence-attention
        relation_repre = tf.keras.layers.Embedding(Config.model.class_num, 3 * Config.model.cnn_filters)(labels)
        r = tf.expand_dims(relation_repre, -1)  # [pair_num,dim,1]
        score = tf.squeeze(tf.matmul(sen_matrix, r), -1)  # [pair_num,max_pair_count]
        log_score = tf.exp(score - tf.reduce_max(score, -1, True)) * tf.cast(mask, tf.float32)
        softmax = log_score / (tf.reduce_sum(log_score, -1, True) + 1e-8)
        alpha = tf.expand_dims(softmax, -1)  # [pair_num,max_pair_count,1]
        sen_att = tf.reduce_sum(alpha * sen_matrix, 1)  # [pair_num, dim]
        return sen_att, labels
