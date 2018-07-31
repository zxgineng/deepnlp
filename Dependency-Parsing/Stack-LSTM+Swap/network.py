import tensorflow as tf

from utils import Config
from data_loader import load_pretrained_vec


class Graph(tf.keras.Model):
    def __init__(self):
        super(Graph, self).__init__()
        self._build_layers()

    def _build_layers(self):
        self.p_embedding = tf.keras.layers.Embedding(Config.model.pos_num, Config.model.pos_embedding_size,
                                                     name='pos_embedding')
        self.rel_embedding = tf.keras.layers.Embedding(Config.model.dep_num * 2,
                                                       Config.model.comp_action_embedding_size, name='rel_embedding')
        self.history_a_embedding = tf.keras.layers.Embedding(3 + Config.model.dep_num * 4,
                                                             Config.model.history_action_embedding_size,
                                                             name='action_embedding')
        wordvec = load_pretrained_vec()
        self.w_embedding = tf.keras.layers.Embedding(wordvec.shape[0], wordvec.shape[1],
                                                     tf.constant_initializer(wordvec, tf.float32),
                                                     name='word_embedding')
        self.embedding_dense = tf.keras.layers.Dense(Config.model.embedding_fc_unit, tf.nn.relu, name='embedding_fc')
        self.recurse_dense = tf.keras.layers.Dense(Config.model.embedding_fc_unit, tf.nn.tanh, name='recurse_fc')

        lstm_cells = [tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer())
                      for _ in range(Config.model.lstm_layer_num)]
        self.stack_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        lstm_cells = [tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer())
                      for _ in range(Config.model.lstm_layer_num)]
        self.buff_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        lstm_cells = [tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer())
                      for _ in range(Config.model.lstm_layer_num)]
        self.action_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        self.final_dense = tf.keras.layers.Dense(2 + 2 * Config.model.dep_num, name='softmax_fc')

    def _embedding(self, tree_word_id, tree_pos_id, buff_word_id, buff_pos_id, comp_rel_id, history_action_id):

        buff_pos_embedded = self.p_embedding(buff_pos_id)
        tree_pos_embedded = self.p_embedding(tree_pos_id)
        comp_rel_embedded = self.rel_embedding(comp_rel_id)
        history_action_embedded = self.history_a_embedding(history_action_id)
        buff_word_embedded = self.w_embedding(buff_word_id)
        tree_word_embedded = self.w_embedding(tree_word_id)
        buff_embedded = tf.concat([buff_word_embedded, buff_pos_embedded], -1)
        tree_embedded = tf.concat([tree_word_embedded, tree_pos_embedded], -1)
        # learned word fc
        buff_embedded = self.embedding_dense(buff_embedded)
        tree_embedded = self.embedding_dense(tree_embedded)

        return tree_embedded, buff_embedded, comp_rel_embedded, history_action_embedded

    def _recursiveNN(self, tree_embedded, comp_rel_embedded, comp_head_order, comp_dep_order, is_leaf, stack_order):
        recursive_tensors = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        recursive_tensors = recursive_tensors.write(0, tree_embedded[:, 0])

        combine_composition = lambda head, rel, dep: self.recurse_dense(tf.concat([head, rel, dep], -1))

        batch_size = tf.shape(is_leaf, out_type=tf.int64)[0]
        tree_size = tf.shape(is_leaf)[1]

        def body(i, recursive_tensors):
            temp = recursive_tensors.stack()  # [i,batch_size]
            node_is_leaf = tf.cast(is_leaf[:, i], tf.bool)
            head = tf.gather_nd(temp, tf.stack([comp_head_order[:, i], tf.range(batch_size, dtype=tf.int64)], -1))
            rel = comp_rel_embedded[:, i]
            dep = tf.gather_nd(temp, tf.stack([comp_dep_order[:, i], tf.range(batch_size, dtype=tf.int64)], -1))
            origin_embedded = tree_embedded[:, i]
            combine_embedded = combine_composition(head, rel, dep)
            node_tensor = tf.where(node_is_leaf, origin_embedded, combine_embedded)
            recursive_tensors = recursive_tensors.write(i, node_tensor)
            i = tf.add(i, 1)

            return i, recursive_tensors

        def cond(i, recursive_tensors):
            return tf.less(i, tree_size)

        [_, recursive_tensors] = tf.while_loop(cond, body, [1, recursive_tensors])

        result = recursive_tensors.stack()
        # clear TensorArray to avoid size error in next sess.run(size is fixed after first run even dynamic_size=True)
        recursive_tensors.close()

        stack_embedded = tf.gather_nd(result, tf.stack(
            [stack_order, tf.ones_like(stack_order) * tf.reshape(tf.range(batch_size, dtype=tf.int64), [-1, 1])], -1))

        return stack_embedded

    def call(self, inputs, mode):
        if mode == 'train':
            self.is_training = True
        else:
            self.is_training = False

        tree_word_id = inputs['tree_word_id']
        tree_pos_id = inputs['tree_pos_id']
        buff_word_id = inputs['buff_word_id']
        buff_pos_id = inputs['buff_pos_id']
        history_action_id = inputs['history_action_id']
        comp_head_order = inputs['comp_head_order']
        comp_dep_order = inputs['comp_dep_order']
        comp_rel_id = inputs['comp_rel_id']
        is_leaf = inputs['is_leaf']
        stack_order = inputs['stack_order']
        stack_length = inputs['stack_length']
        buff_length = inputs['buff_length']
        history_action_length = inputs['history_action_length']

        tree_embedded, buff_embedded, comp_rel_embedded, history_action_embedded = self._embedding(tree_word_id,
                                                                                                   tree_pos_id,
                                                                                                   buff_word_id,
                                                                                                   buff_pos_id,
                                                                                                   comp_rel_id,
                                                                                                   history_action_id)

        stack_embedded = self._recursiveNN(tree_embedded, comp_rel_embedded, comp_head_order, comp_dep_order, is_leaf,
                                           stack_order)
        with tf.variable_scope('stack_lstm'):
            outputs, state = tf.nn.dynamic_rnn(self.stack_lstm, stack_embedded, stack_length, dtype=tf.float32)
            stack_lstm_outputs = state[1].h
        with tf.variable_scope('buff_lstm'):
            outputs, state = tf.nn.dynamic_rnn(self.buff_lstm, buff_embedded, buff_length, dtype=tf.float32)
            buff_lstm_outputs = state[1].h
        with tf.variable_scope('action_lstm'):
            outputs, state = tf.nn.dynamic_rnn(self.action_lstm, history_action_embedded, history_action_length,
                                               dtype=tf.float32)
            action_lstm_outputs = state[1].h

        logits = self.final_dense(tf.concat([stack_lstm_outputs, buff_lstm_outputs, action_lstm_outputs], -1))

        return logits
