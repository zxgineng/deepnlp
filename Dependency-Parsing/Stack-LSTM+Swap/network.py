import tensorflow as tf

from utils import Config
from data_loader import load_pretrained_vec


class Graph(tf.keras.Model):
    def __init__(self):
        super(Graph, self).__init__()
        self._build_layers()

    def _build_layers(self):
        self.p_embedding = tf.keras.layers.Embedding(Config.model.pos_num, Config.model.pos_embedding_size)
        self.comp_a_embedding = tf.keras.layers.Embedding(Config.model.dep_num * 2,
                                                          Config.model.comp_action_embedding_size)
        self.history_a_embedding = tf.keras.layers.Embedding(3 + Config.model.dep_num * 4,
                                                             Config.model.history_action_embedding_size)
        wordvec = load_pretrained_vec()
        self.w_embedding = tf.keras.layers.Embedding(wordvec.shape[0], wordvec.shape[1],
                                                     tf.constant_initializer(wordvec, tf.float32))
        self.embedding_dense = tf.keras.layers.Dense(Config.model.embedding_fc_unit, tf.nn.relu)
        self.recurse_dense = tf.keras.layers.Dense(Config.model.embedding_fc_unit, tf.nn.tanh)

        lstm_cells = [tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer())
                      for _ in range(Config.model.lstm_layer_num)]
        self.stack_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        lstm_cells = [tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer())
                      for _ in range(Config.model.lstm_layer_num)]
        self.buff_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        lstm_cells = [tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer())
                      for _ in range(Config.model.lstm_layer_num)]
        self.action_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        self.final_dense = tf.keras.layers.Dense(2 + 2 * Config.model.dep_num)

    def _embedding(self, buff_word_id, buff_pos_id, comp_word_id, comp_pos_id, comp_action_id, comp_action_len,
                   history_action_id):
        buff_pos_embedded = self.p_embedding(buff_pos_id)
        comp_pos_embedded = self.p_embedding(comp_pos_id)
        comp_action_embedded = self.comp_a_embedding(comp_action_id)
        history_action_embedded = self.history_a_embedding(history_action_id)
        buff_word_embedded = self.w_embedding(buff_word_id)
        # [batch_size,max(stack_num),max(comp_word_num),embedding_size]
        comp_word_embedded = self.w_embedding(comp_word_id)
        buff_embedded = tf.concat([buff_word_embedded, buff_pos_embedded], -1)
        comp_embedded = tf.concat([comp_word_embedded, comp_pos_embedded], -1)
        # learned word fc
        buff_embedded = self.embedding_dense(buff_embedded)
        comp_embedded = self.embedding_dense(comp_embedded)
        # composition function
        new_batch_stack_embedded = []
        for i, l in enumerate(comp_action_len):  # loop through all samples
            new_sample_stack_embedded = []
            for j, a_l in enumerate(l):  # loop through stack word
                embedding_dict = {}
                if int(a_l) == 0:
                    new_sample_stack_embedded.append(comp_embedded[i][j][0])
                else:
                    for n in range(int(a_l)):  # get recursive embedding
                        head_id = int(comp_word_id[i, j, 2 * n])
                        dep_id = int(comp_word_id[i, j, 2 * n + 1])
                        rel_e = comp_action_embedded[i, j, n]
                        head_e = comp_embedded[i, j, 2 * n] if head_id not in embedding_dict else embedding_dict[
                            head_id]
                        dep_e = comp_embedded[i, j, 2 * n + 1] if dep_id not in embedding_dict else embedding_dict[
                            dep_id]
                        new_embedded = self.recurse_dense(tf.expand_dims(tf.concat([head_e, rel_e, dep_e], -1), 0))[0]
                        embedding_dict[head_id] = new_embedded
                    new_sample_stack_embedded.append(new_embedded)
            new_batch_stack_embedded.append(tf.stack(new_sample_stack_embedded))
        stack_embedded = tf.stack(new_batch_stack_embedded)

        return stack_embedded, buff_embedded, history_action_embedded

    def call(self, inputs, mode):
        if mode == 'train':
            is_training = True
        else:
            is_training = False

        buff_word_id = inputs['buff_word_id']
        buff_pos_id = inputs['buff_pos_id']
        history_action_id = inputs['history_action_id']
        comp_word_id = inputs['comp_word_id']  # [batch_size,max(stack_num),max(comp_word_num)]
        comp_pos_id = inputs['comp_pos_id']  # [batch_size,max(stack_num),max(comp_word_num)]
        comp_action_id = inputs['comp_action_id']  # [batch_size,max(stack_num),max(comp_action_num)]
        comp_action_len = inputs['comp_action_len']  # [batch_size,max(stack_num)]
        stack_length = inputs['stack_length']
        buff_length = inputs['buff_length']
        history_action_length = inputs['history_action_length']

        stack_embedded, buff_embedded, history_action_embedded = self._embedding(buff_word_id, buff_pos_id,
                                                                                 comp_word_id, comp_pos_id,
                                                                                 comp_action_id, comp_action_len,
                                                                                 history_action_id)

        outputs, state = tf.nn.dynamic_rnn(self.stack_lstm, stack_embedded, stack_length, dtype=tf.float32)
        stack_lstm_outputs = state[1].h
        outputs, state = tf.nn.dynamic_rnn(self.buff_lstm, buff_embedded, buff_length, dtype=tf.float32)
        buff_lstm_outputs = state[1].h
        outputs, state = tf.nn.dynamic_rnn(self.action_lstm, history_action_embedded, history_action_length,
                                           dtype=tf.float32)
        action_lstm_outputs = state[1].h

        logits = self.final_dense(tf.concat([stack_lstm_outputs, buff_lstm_outputs, action_lstm_outputs], -1))

        return logits
