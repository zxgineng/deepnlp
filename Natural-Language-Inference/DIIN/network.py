import tensorflow as tf

from utils import Config
from data_loader import load_pretrained_vec


class Graph(tf.keras.Model):
    def __init__(self):
        super(Graph, self).__init__()

        self.embedding_layer = Embedding_Layer(name='embedding_layer')
        self.p_encoding_layer = Encoding_layer(name='p_encoding_layer')
        self.h_encoding_layer = Encoding_layer(name='h_encoding_layer')
        self.interaction_layer = Interaction_Layer(name='interaction_layer')
        self.feature_extraction_layer = Feature_Extraction_Layer(name='feature_extraction_layer')
        self.softmax_dense = tf.keras.layers.Dense(2, name='softmax_dense')

    def call(self, inputs, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
        else:
            is_training = False

        p_word_id = inputs['p_word_id']
        p_pos_id = inputs['p_pos_id']
        p_char_images = inputs['p_char_images']
        h_word_id = inputs['h_word_id']
        h_pos_id = inputs['h_pos_id']
        h_char_images = inputs['h_char_images']

        p_embedded = self.embedding_layer(p_word_id, p_pos_id, p_char_images, is_training)
        h_embedded = self.embedding_layer(h_word_id, h_pos_id, h_char_images, is_training)
        p_encoded = self.p_encoding_layer(p_embedded, is_training)
        h_encoded = self.h_encoding_layer(h_embedded, is_training)
        net = self.interaction_layer(p_encoded, h_encoded)
        net = self.feature_extraction_layer(net)
        net = tf.keras.layers.Flatten()(net)
        logits = self.softmax_dense(net)
        return logits


class Dropout_Dense(tf.keras.layers.Dense):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Dropout_Dense, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.dropout = tf.keras.layers.Dropout(1 - tf.pow(Config.train.dropout_decay, Config.train.epoch))

    def call(self, inputs, training):
        inputs = self.dropout(inputs, training=training)
        outputs = super(Dropout_Dense, self).call(inputs)
        return outputs


class Embedding_Layer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Embedding_Layer, self).__init__(**kwargs)

        self.dropout = tf.keras.layers.Dropout(1 - tf.pow(Config.train.dropout_decay, Config.train.epoch))
        wordvec = load_pretrained_vec()
        self.word_embedding = tf.keras.layers.Embedding(wordvec.shape[0], wordvec.shape[1],
                                                        tf.constant_initializer(wordvec, tf.float32),
                                                        name='word_embedding')
        self.pos_embedding = tf.keras.layers.Embedding(Config.model.pos_num, Config.model.pos_embedding_size,
                                                       name='pos_embedding')
        self.conv3d_0 = tf.keras.layers.Conv3D(16, [1, 3, 3], padding='SAME', activation=tf.nn.relu, name='conv3d_0')
        self.conv3d_1 = tf.keras.layers.Conv3D(16, [1, 3, 3], padding='SAME', activation=tf.nn.relu, name='conv3d_1')
        self.maxpool3d_0 = tf.keras.layers.MaxPool3D([1, 1, 2], [1, 1, 2], 'SAME')  # pool width only
        self.maxpool3d_1 = tf.keras.layers.MaxPool3D([1, 2, 2], [1, 2, 2], 'SAME')
        self.conv3d_2 = tf.keras.layers.Conv3D(32, [1, 3, 3], padding='SAME', activation=tf.nn.relu, name='conv3d_2')
        self.conv3d_3 = tf.keras.layers.Conv3D(32, [1, 3, 3], padding='SAME', activation=tf.nn.relu, name='conv3d_3')
        self.conv3d_4 = tf.keras.layers.Conv3D(64, [1, 3, 3], padding='SAME', activation=tf.nn.relu, name='conv3d_4')
        self.conv3d_5 = tf.keras.layers.Conv3D(64, [1, 3, 3], padding='SAME', activation=tf.nn.relu, name='conv3d_5')
        self.cnn_dense = tf.keras.layers.Dense(Config.model.cnn_dense_units, tf.nn.relu, name='cnn_dense')

    def _cnn_embedding(self, inputs):
        net = self.conv3d_0(inputs)
        net = self.conv3d_1(net)
        net = self.maxpool3d_1(net)
        net = self.conv3d_2(net)
        net = self.maxpool3d_0(net)
        net = self.conv3d_3(net)
        net = self.maxpool3d_1(net)
        net = self.conv3d_4(net)
        net = self.maxpool3d_0(net)
        net = self.conv3d_5(net)
        net = self.maxpool3d_1(net)
        net = tf.reshape(net, [-1, tf.shape(inputs)[1], tf.size(net[0][0])])
        outputs = self.cnn_dense(net)
        return outputs

    def call(self, word_id, pos_id, char_image, training):
        word_embedded = self.word_embedding(word_id)
        pos_embedded = self.pos_embedding(pos_id)
        word_embedded = self.dropout(word_embedded, training=training)
        pos_embedded = self.dropout(pos_embedded, training=training)
        char_embedded = self._cnn_embedding(char_image)
        outputs = tf.concat([word_embedded, pos_embedded, char_embedded], -1)
        return outputs


class Encoding_layer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Encoding_layer, self).__init__(**kwargs)

        units = Config.model.cnn_dense_units + Config.model.word_embedding_size + Config.model.pos_embedding_size
        self.transfer_dense = {i: Dropout_Dense(units, tf.nn.relu, name='transfer_dense_' + str(i)) for i in
                               range(Config.model.highway_num)}
        self.carry_dense = {i: Dropout_Dense(units, tf.nn.sigmoid, name='carry_dense_' + str(i)) for i in
                            range(Config.model.highway_num)}
        self.a_dense = Dropout_Dense(1, tf.nn.relu, name='alpha_dense')
        self.fuse_z_dense = Dropout_Dense(units, tf.nn.tanh, name='fuse_z_dense')
        self.fuse_r_dense = Dropout_Dense(units, tf.nn.sigmoid, name='fuse_r_dense')
        self.fuse_f_dense = Dropout_Dense(units, tf.nn.sigmoid, name='fuse_f_dense')

    def _highway(self, inputs):
        prev = inputs
        cur = None
        for i in range(Config.model.highway_num):
            cur = self.transfer_dense[i](prev, self.is_training) * self.carry_dense[i](prev, self.is_training) + \
                  prev * (1 - self.carry_dense[i](prev, self.is_training))
            prev = cur
        return cur

    def _self_attention(self, inputs):
        part1 = tf.tile(tf.expand_dims(inputs, 2), [1, 1, tf.shape(inputs)[1], 1])  # [B,p,d] -> [B,p,p,d]
        part2 = tf.transpose(part1, [0, 2, 1, 3])
        part3 = tf.multiply(part1, part2)
        A = tf.squeeze(self.a_dense(tf.concat([part1, part2, part3], -1), self.is_training), -1)  # [B,p,p]
        outputs = tf.matmul(tf.nn.softmax(A, -1), inputs)  # [B,p,d]
        return outputs

    def _fuse_gate(self, inputs, attentioned):
        concat = tf.concat([inputs, attentioned], -1)
        z = self.fuse_f_dense(concat, self.is_training)
        r = self.fuse_r_dense(concat, self.is_training)
        f = self.fuse_f_dense(concat, self.is_training)
        outputs = r * inputs + f * z
        return outputs

    def call(self, inputs, training):
        self.is_training = training
        net = self._highway(inputs)
        net = self._self_attention(net)  # [B,p,d]
        net = self._fuse_gate(inputs, net)
        return net


class Interaction_Layer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Interaction_Layer, self).__init__(**kwargs)

    def call(self, premise, hypothesis):
        premise = tf.tile(tf.expand_dims(premise, 2), [1, 1, tf.shape(hypothesis)[1], 1])  # [B,p,d] -> [B,p,h,d]
        hypothesis = tf.tile(tf.expand_dims(hypothesis, 2), [1, 1, tf.shape(premise)[1], 1])  # [B,h,d] -> [B,h,p,d]
        hypothesis = tf.transpose(hypothesis, [0, 2, 1, 3])  # [B,h,p,d] -> [B,p,h,d]
        outputs = premise * hypothesis
        return outputs


class Feature_Extraction_Layer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Feature_Extraction_Layer, self).__init__(**kwargs)

        units = Config.model.cnn_dense_units + Config.model.word_embedding_size + Config.model.pos_embedding_size
        self.scale_conv2d = tf.keras.layers.Conv2D(int(units * Config.model.feature_scale_ratio), 1,
                                                   name='scale_down_conv')
        self.densenet = DenseNet(Config.model.dense_block_num, Config.model.dense_layer_per_block, 20, name='densenet')

    def call(self, inputs):
        net = self.scale_conv2d(inputs)
        outputs = self.densenet(net)
        return outputs


class DenseNet(tf.keras.Model):
    def __init__(self, dense_block_num, dense_layer_per_block, growth_rate, **kwargs):
        super(DenseNet, self).__init__(**kwargs)

        self.dense_block_num = dense_block_num
        self.dense_layer_per_block = dense_layer_per_block
        self.growth_rate = growth_rate

    def _dense_block(self, inputs):
        for i in range(self.dense_layer_per_block):
            outputs = tf.keras.layers.Conv2D(self.growth_rate, [3, 3], padding='SAME', activation=tf.nn.relu)(inputs)
            inputs = tf.concat([inputs, outputs], -1)
        return inputs

    def _transition_block(self, inputs):
        dim = int(inputs.shape.as_list()[-1] * Config.model.dense_transition_ratio)
        net = tf.keras.layers.Conv2D(dim, [1, 1],
                                     padding='SAME', activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.MaxPool2D([2, 2], padding='SAME')(net)
        return outputs

    def call(self, inputs):
        net = inputs
        for i in range(self.dense_block_num):
            net = self._dense_block(net)
            net = self._transition_block(net)
        return net
