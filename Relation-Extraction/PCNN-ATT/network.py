import tensorflow as tf

from utils import Config
from data_loader import load_pretrained_vec


class Graph(tf.keras.Model):
    def __init__(self,**kwargs):
        super(Graph, self).__init__(**kwargs)

        self.embedding_layer = Embedding_Layer(name='embedding_layer')


    def call(self, inputs, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
        else:
            is_training = False


        word_id = inputs['word_id']
        tag_id = inputs['tag_id']
        pos_1 = inputs['pos_1']
        pos_2 = inputs['pos_2']


        embedded = self.embedding_layer(word_id,pos_1,pos_2)



        return sen_vec  # [B,seq_len,size]




class Embedding_Layer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Embedding_Layer, self).__init__(**kwargs)

        wordvec = load_pretrained_vec()
        self.word_embedding = tf.keras.layers.Embedding(wordvec.shape[0], wordvec.shape[1],
                                                        tf.constant_initializer(wordvec, tf.float32),
                                                        name='word_embedding')
        self.tag_embedding = tf.keras.layers.Embedding(Config.model.tag_num,Config.model.tag_embedding_size,
                                                       name='tag_embedding')
        self.pos_embedding = tf.keras.layers.Embedding()
