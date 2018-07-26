import tensorflow as tf

from utils import Config
from data_loader import load_pretrained_vec


class Graph:
    def __init__(self, mode):
        self.mode = mode
        if mode == 'train':
            self.is_training = True
        else:
            self.is_training = False

    def build(self, inputs):
        buff_word_id = inputs['buff_word_id']
        buff_pos_id = inputs['buff_pos_id']
        history_action_id = inputs['history_action_id']
        comp_word_id = inputs['comp_word_id']  # [batch_size,max(stack_num),max(comp_word_num)]
        comp_pos_id = inputs['comp_pos_id']  # [batch_size,max(stack_num),max(comp_word_num)]
        comp_action_id =inputs['comp_action_id']  # [batch_size,max(stack_num),max(comp_action_num)]
        comp_action_len =inputs['comp_action_len']  # [batch_size,max(stack_num)]
        stack_length = inputs['stack_length']
        buff_length = inputs['buff_length']
        history_action_length = inputs['history_action_length']

        stack_embedded, buff_embedded, history_action_embedded = self._embedding(buff_word_id,buff_pos_id,comp_word_id,comp_pos_id,comp_action_id,comp_action_len,history_action_id)
        stack_lstm_outputs = self._lstm(stack_embedded,stack_length)
        buff_lstm_outputs = self._lstm(buff_embedded,buff_length)
        action_lstm_outputs = self._lstm(history_action_embedded,history_action_length)

        logits = self._fc_layer(tf.concat([stack_lstm_outputs,buff_lstm_outputs,action_lstm_outputs],-1))

        return logits

    def _embedding(self, buff_word_id,buff_pos_id,comp_word_id,comp_pos_id,comp_action_id,comp_action_len,history_action_id):

        p_embedding = tf.keras.layers.Embedding(Config.model.pos_num, Config.model.pos_embedding_size)
        buff_pos_embedded = p_embedding(buff_pos_id)
        comp_pos_embedded = p_embedding(comp_pos_id)

        comp_a_embedding = tf.keras.layers.Embedding(Config.model.dep_num * 2, Config.model.comp_action_embedding_size)
        comp_action_embedded = comp_a_embedding(comp_action_id)

        history_a_embedding = tf.keras.layers.Embedding(3 + Config.model.dep_num * 4, Config.model.history_action_embedding_size)
        history_action_embedded = history_a_embedding(history_action_id)

        wordvec = load_pretrained_vec()
        w_embedding = tf.keras.layers.Embedding(wordvec.shape[0], wordvec.shape[1],
                                                tf.constant_initializer(wordvec, tf.float32))
        buff_word_embedded = w_embedding(buff_word_id)
        comp_word_embedded = w_embedding(comp_word_id)  # [batch_size,max(stack_num),max(comp_word_num),embedding_size]

        buff_embedded = tf.concat([buff_word_embedded, buff_pos_embedded], -1)
        comp_embedded = tf.concat([comp_word_embedded, comp_pos_embedded], -1)

        # learned word fc
        embedding_dense = tf.keras.layers.Dense(Config.model.embedding_fc_unit, tf.nn.relu)
        buff_embedded = embedding_dense(buff_embedded)
        comp_embedded = embedding_dense(comp_embedded)

        recurse_dense = tf.keras.layers.Dense(Config.model.embedding_fc_unit, tf.nn.tanh)

        # composition function
        new_batch_stack_embedded = []
        for i, l in enumerate(comp_action_len):  # loop through all samples
            new_sample_stack_embedded = []
            for j,a_l in enumerate(l):  # loop through stack word
                embedding_dict = {}
                if int(a_l) == 0:
                    new_sample_stack_embedded.append(comp_embedded[i][j][0])
                else:
                    for n in range(int(a_l)):  # get recursive embedding
                        head_id = int(comp_word_id[i,j,2*n])
                        dep_id = int(comp_word_id[i,j,2*n + 1])
                        rel_e = comp_action_embedded[i,j,n]
                        head_e = comp_embedded[i,j,2*n] if head_id not in embedding_dict else embedding_dict[head_id]
                        dep_e = comp_embedded[i,j,2*n+ 1] if dep_id not in embedding_dict else embedding_dict[dep_id]
                        new_embedded = recurse_dense(tf.expand_dims(tf.concat([head_e, rel_e, dep_e], -1), 0))[0]
                        embedding_dict[head_id] = new_embedded
                    new_sample_stack_embedded.append(new_embedded)
            new_batch_stack_embedded.append(tf.stack(new_sample_stack_embedded))
        stack_embedded = tf.stack(new_batch_stack_embedded)

        return stack_embedded, buff_embedded, history_action_embedded

    def _lstm(self, inputs, length):
        lstm_cells = [tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer())
                      for _ in range(Config.model.lstm_layer_num)]
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs, length, dtype=tf.float32)

        return state[1].h

    def _fc_layer(self, inputs):
        logits = tf.keras.layers.Dense(2 + 2*Config.model.dep_num)(inputs)
        return logits
