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

        self.history_a_embedding = tf.keras.layers.Embedding(4 + Config.model.dep_num * 4,
                                                             Config.model.history_action_embedding_size,
                                                             name='action_embedding')
        wordvec = load_pretrained_vec()
        self.w_embedding = tf.keras.layers.Embedding(wordvec.shape[0], wordvec.shape[1],
                                                     tf.constant_initializer(wordvec, tf.float32),
                                                     name='word_embedding')

        self.learned_word_dense = tf.keras.layers.Dense(Config.model.embedding_fc_unit, tf.nn.relu,
                                                        name='learned_word_fc')
        self.tree_lstm_cell = TreeLSTMCell(Config.model.lstm_unit)

        lstm_cells = [tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer())
                      for _ in range(2)]
        self.stack_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        self.fw_lstm_cell0 = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer(),
                                                     name='fw_cell0')
        self.bw_lstm_cell0 = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer(),
                                                     name='bw_cell0')
        self.fw_lstm_cell1 = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer(),
                                                     name='fw_cell1')
        self.bw_lstm_cell1 = tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer(),
                                                     name='bw_cell1')

        lstm_cells = [tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer())
                      for _ in range(2)]
        self.action_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        lstm_cells = [tf.nn.rnn_cell.LSTMCell(Config.model.lstm_unit, initializer=tf.orthogonal_initializer())
                      for _ in range(2)]
        self.deque_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        self.final_dense = tf.keras.layers.Dense(4 + 4 * Config.model.dep_num, tf.nn.relu, name='softmax_fc')

    def _embedding(self, tree_word_id, tree_pos_id, token_word_id, token_pos_id, history_action_id, deque_word_id,
                   deque_pos_id):

        token_pos_embedded = self.p_embedding(token_pos_id)
        tree_pos_embedded = self.p_embedding(tree_pos_id)
        deque_pos_embedded = self.p_embedding(deque_pos_id)

        history_action_embedded = self.history_a_embedding(history_action_id)

        token_word_embedded = self.w_embedding(token_word_id)
        tree_word_embedded = self.w_embedding(tree_word_id)
        deque_word_embedded = self.w_embedding(deque_word_id)

        token_embedded = tf.concat([token_word_embedded, token_pos_embedded], -1)
        tree_embedded = tf.concat([tree_word_embedded, tree_pos_embedded], -1)
        deque_embedded = tf.concat([deque_word_embedded, deque_pos_embedded], -1)

        token_embedded = self.learned_word_dense(token_embedded)
        tree_embedded = self.learned_word_dense(tree_embedded)
        deque_embedded = self.learned_word_dense(deque_embedded)

        return tree_embedded, token_embedded, deque_embedded, history_action_embedded

    def _dynamic_tree_lstm(self, tree_embedded, children_order, stack_order):
        h_tensors = tf.TensorArray(tf.float32, size=200, dynamic_size=True, clear_after_read=False)
        c_tensors = tf.TensorArray(tf.float32, size=200, dynamic_size=True, clear_after_read=False)

        batch_size = tf.shape(children_order, out_type=tf.int64)[0]
        children_num = tf.shape(children_order)[1]

        h_tensors = h_tensors.write(0, tf.zeros(
            tf.stack([batch_size, Config.model.lstm_unit])))  # idx 0 represents no children states
        c_tensors = c_tensors.write(0, tf.zeros(tf.stack([batch_size, Config.model.lstm_unit])))

        tree_lstm_one_step = lambda inputs, c, h: self.tree_lstm_cell(inputs, tf.nn.rnn_cell.LSTMStateTuple(c, h))

        def body(i, h_tensors, c_tensors):
            h_temp = h_tensors.gather(tf.range(i + 1))
            c_temp = c_tensors.gather(tf.range(i + 1))

            indices = tf.stack([children_order[:, i], tf.ones_like(children_order[:, i]) * tf.expand_dims(
                tf.range(batch_size, dtype=tf.int64), -1)], -1)

            h = tf.gather_nd(h_temp, indices)
            c = tf.gather_nd(c_temp, indices)

            output, state = tree_lstm_one_step(tree_embedded[:, i], c, h)
            h_tensors.write(i + 1, state.h)
            c_tensors.write(i + 1, state.c)

            i = tf.add(i, 1)

            return i, h_tensors, c_tensors

        def cond(i, h_tensors, c_tensors):
            return tf.less(i, children_num)

        [_, h_tensors, c_tensors] = tf.while_loop(cond, body, [0, h_tensors, c_tensors])

        result = h_tensors.stack()

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
        token_word_id = inputs['token_word_id']
        token_pos_id = inputs['token_pos_id']
        history_action_id = inputs['history_action_id']
        buff_top_id = inputs['buff_top_id']
        deque_word_id = inputs['deque_word_id']
        deque_pos_id = inputs['deque_pos_id']
        deque_length = inputs['deque_length']
        children_order = inputs['children_order']
        stack_order = inputs['stack_order']
        stack_length = inputs['stack_length']
        token_length = inputs['token_length']
        history_action_length = inputs['history_action_length']

        tree_embedded, token_embedded, deque_embedded, history_action_embedded = \
            self._embedding(tree_word_id, tree_pos_id, token_word_id, token_pos_id, history_action_id,
                            deque_word_id, deque_pos_id)

        with tf.variable_scope('tree_lstm'):
            stack_embedded = self._dynamic_tree_lstm(tree_embedded, children_order, stack_order)

        with tf.variable_scope('stack_lstm'):
            outputs, states = tf.nn.dynamic_rnn(self.stack_lstm, stack_embedded, stack_length, dtype=tf.float32)
            stack_lstm_outputs = states[1].h

        with tf.variable_scope('buff_lstm'):
            outputs, states = tf.nn.bidirectional_dynamic_rnn(self.fw_lstm_cell0, self.bw_lstm_cell0, token_embedded,
                                                              token_length, dtype=tf.float32)
            outputs = tf.concat(outputs, -1)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(self.fw_lstm_cell1, self.bw_lstm_cell1, outputs,
                                                              token_length, dtype=tf.float32)
            fw_outputs, bw_outputs = outputs
            batch_size = tf.shape(stack_length, out_type=tf.int64)[0]
            fw_top = tf.gather_nd(fw_outputs, tf.stack([tf.range(batch_size, dtype=tf.int64), buff_top_id], -1))
            fw_bottom = states[0].h
            bw_top = tf.gather_nd(bw_outputs, tf.stack([tf.range(batch_size, dtype=tf.int64), buff_top_id], -1))
            bw_bottom = tf.gather_nd(bw_outputs, tf.stack([tf.range(batch_size, dtype=tf.int64), token_length - 1], -1))
            buff_lstm_outputs = tf.concat([(fw_bottom - fw_top), (bw_top - bw_bottom)], -1)

        with tf.variable_scope('action_lstm'):
            outputs, states = tf.nn.dynamic_rnn(self.action_lstm, history_action_embedded, history_action_length,
                                                dtype=tf.float32)
            action_lstm_outputs = states[1].h

        with tf.variable_scope('deque_lstm'):
            outputs, states = tf.nn.dynamic_rnn(self.deque_lstm, deque_embedded, deque_length, dtype=tf.float32)
            deque_lstm_outputs = states[1].h

        logits = self.final_dense(
            tf.concat([stack_lstm_outputs, buff_lstm_outputs, action_lstm_outputs, deque_lstm_outputs], -1))

        return logits


class TreeLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    def __init__(self, num_units):
        super(TreeLSTMCell, self).__init__(num_units=num_units)
        self._num_units = num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernel = self.add_variable("kernel",
                                         shape=[input_depth + h_depth, 4 * self._num_units],
                                         initializer=tf.orthogonal_initializer())
        self._bias = self.add_variable("bias",
                                       shape=[4 * self._num_units],
                                       initializer=tf.zeros_initializer(dtype=self.dtype))

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        self._ijo_kernel = self._kernel[:, :3 * self._num_units]
        self._ijo_bias = self._bias[:3 * self._num_units]
        self._f_kernel = self._kernel[:, 3 * self._num_units:]
        self._f_bias = self._bias[3 * self._num_units:]

        self.built = True

    def call(self, inputs, state):
        c, h = state  # [batch_size, children_num, num_units]
        hj = tf.reduce_sum(h, 1)  # [batch_size,num_units]

        ijo = tf.matmul(tf.concat([inputs, hj], -1), self._ijo_kernel)
        ijo = tf.nn.bias_add(ijo, self._ijo_bias)

        f = tf.tensordot(tf.concat([tf.tile(tf.expand_dims(inputs, 1), [1, tf.shape(h)[1], 1]), h], -1),
                         self._f_kernel, [[2], [0]])
        f = tf.nn.bias_add(f, self._f_bias)  # [batch_size, children_num, num_units]

        forget_bias_tensor = tf.constant(self._forget_bias, dtype=tf.float32)

        i, j, o = tf.split(value=ijo, num_or_size_splits=3, axis=1)

        new_c = tf.add(tf.reduce_sum(tf.multiply(c, tf.sigmoid(tf.add(f, forget_bias_tensor))), 1),
                       tf.multiply(tf.sigmoid(i), self._activation(j)))

        new_h = tf.multiply(self._activation(new_c), tf.sigmoid(o))
        new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

        return new_h, new_state
