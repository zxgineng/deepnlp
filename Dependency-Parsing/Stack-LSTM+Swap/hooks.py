import numpy as np
import tensorflow as tf
import os

from utils import Config
from data_loader import Token, Sentence, ArcStandardParser,id2word,id2pos,load_vocab,load_pos


class EvalHook(tf.train.SessionRunHook):
    def __init__(self, model_dir=None):
        self.total = 0
        self.correct_head = 0
        self.correct_dep = 0
        self.parser = ArcStandardParser()
        if model_dir:
            self.model_dir = os.path.join(Config.train.model_dir, 'eval_' + model_dir)
        else:
            self.model_dir = os.path.join(Config.train.model_dir, 'eval')

    def begin(self):
        self._summary_writer = tf.summary.FileWriter(self.model_dir)

    def before_run(self, run_context):

        return tf.train.SessionRunArgs(
            ['next_batch:0', 'next_batch:1', 'next_batch:4', 'next_batch:2', 'next_batch:3'])

    def after_run(self, run_context, run_values):
        word, pos, length, head, dep_id = run_values.results
        batch_size = word.shape[0]
        total_sen = [Sentence([Token(n + 1, word[i][n].decode(), pos[i][n].decode(), None, None) for
                               n in range(length[i])]) for i in range(batch_size)]

        while True:
            batch_tree_word_id, batch_tree_pos_id, batch_buff_word_id, batch_buff_pos_id, batch_history_action_id, \
            batch_comp_head_order, batch_comp_dep_order, batch_comp_rel_id, batch_is_leaf, batch_stack_order, \
            batch_stack_length, batch_buff_length, batch_history_action_length = \
                [], [], [], [], [], [], [], [], [], [], [], [], []

            for sen in total_sen:
                if not sen.terminate:
                    tree_word_id, tree_pos_id, buff_word_id, buff_pos_id, history_action_id, comp_head_order, \
                    comp_dep_order, comp_rel_id, is_leaf, stack_order = self.parser.extract_from_current_state(sen)

                    batch_tree_word_id.append(tree_word_id)
                    batch_tree_pos_id.append(tree_pos_id)
                    batch_buff_word_id.append(buff_word_id)
                    batch_buff_pos_id.append(buff_pos_id)
                    batch_history_action_id.append(history_action_id)
                    batch_comp_head_order.append(comp_head_order)
                    batch_comp_dep_order.append(comp_dep_order)
                    batch_comp_rel_id.append(comp_rel_id)
                    batch_is_leaf.append(is_leaf)
                    batch_stack_order.append(stack_order)
                    batch_stack_length.append(len(stack_order))
                    batch_buff_length.append(len(buff_word_id))
                    batch_history_action_length.append(len(history_action_id))

            batch_tree_word_id = tf.keras.preprocessing.sequence.pad_sequences(batch_tree_word_id,
                                                                               dtype='int64', padding='post')
            batch_tree_pos_id = tf.keras.preprocessing.sequence.pad_sequences(batch_tree_pos_id,
                                                                              dtype='int64', padding='post')
            batch_buff_word_id = tf.keras.preprocessing.sequence.pad_sequences(batch_buff_word_id,
                                                                               dtype='int64', padding='post')
            batch_buff_pos_id = tf.keras.preprocessing.sequence.pad_sequences(batch_buff_pos_id,
                                                                              dtype='int64', padding='post')
            batch_history_action_id = tf.keras.preprocessing.sequence.pad_sequences(batch_history_action_id,
                                                                                    dtype='int64', padding='post')
            batch_comp_head_order = tf.keras.preprocessing.sequence.pad_sequences(batch_comp_head_order,
                                                                                  dtype='int64', padding='post')
            batch_comp_dep_order = tf.keras.preprocessing.sequence.pad_sequences(batch_comp_dep_order,
                                                                                 dtype='int64', padding='post')
            batch_comp_rel_id = tf.keras.preprocessing.sequence.pad_sequences(batch_comp_rel_id,
                                                                              dtype='int64', padding='post')
            batch_is_leaf = tf.keras.preprocessing.sequence.pad_sequences(batch_is_leaf,
                                                                          dtype='int64', padding='post')
            batch_stack_order = tf.keras.preprocessing.sequence.pad_sequences(batch_stack_order,
                                                                              dtype='int64', padding='post')

            feed_dict = {'tree_word_id:0': batch_tree_word_id, 'tree_pos_id:0': batch_tree_pos_id,
                         'buff_word_id:0': batch_buff_word_id,'buff_pos_id:0': batch_buff_pos_id,
                         'history_action_id:0': batch_history_action_id,'comp_head_order:0': batch_comp_head_order,
                         'comp_dep_order:0': batch_comp_dep_order,'comp_rel_id:0': batch_comp_rel_id,
                         'is_leaf:0': batch_is_leaf,'stack_order:0': batch_stack_order,
                         'stack_length:0': batch_stack_length,'buff_length:0': batch_buff_length,
                         'history_action_length:0': batch_history_action_length}

            prob = run_context.session.run("Softmax:0", feed_dict)


            idx = 0
            for sen in total_sen:
                if not sen.terminate:
                    legal_transitions = self.parser.get_legal_transitions(sen)
                    transition = np.argmax(np.array(legal_transitions) * prob[idx])
                    if transition not in [0, 1]:
                        self.parser.update_composition(sen, transition)  # update composition
                    self.parser.update_state_by_transition(sen, transition)  # update stack and buff
                    idx += 1
                    if self.parser.terminal(sen):
                        sen.terminate = True
            if [sen.terminate for sen in total_sen] == [True] * batch_size:
                for i, sen in enumerate(total_sen):
                    gold_head = head[i, :length[i]]
                    pred_head = np.array([t.pred_head_id for t in sen.tokens])
                    assert len(gold_head) == len(pred_head)
                    correct_head = gold_head == pred_head
                    self.correct_head += np.sum(correct_head)
                    gold_dep = dep_id[i, :length[i]]
                    pred_dep = np.array([t.pred_dep_id for t in sen.tokens])
                    correct_dep = gold_dep == pred_dep
                    self.correct_dep += np.sum(np.logical_and(correct_head, correct_dep))
                    self.total += length[i]
                break

    def end(self, session):

        global_step = session.run(tf.train.get_global_step())
        summary_op = tf.get_default_graph().get_collection('acc')

        head_acc = self.correct_head / self.total
        dep_acc = self.correct_dep / self.total

        summary = session.run(summary_op, {'head_ph:0': head_acc,
                                           'dep_ph:0': dep_acc})
        for s in summary:
            self._summary_writer.add_summary(s, global_step)

        self._summary_writer.flush()

        print('*' * 40)
        print("epoch", Config.train.epoch + 1, 'finished')
        print('UAS:', head_acc, '    LAS:', dep_acc)
        print('*' * 40)


class PredHook(tf.train.SessionRunHook):
    def __init__(self):
        self.parser = ArcStandardParser()
        self.vocab = load_vocab()
        self.pos_dict = load_pos()
        self.pred_head = []
        self.pred_dep = []
        self.no_op = tf.no_op()

    def before_run(self, run_context):
        length,pos_id,word_id = run_context.session.run(['fifo_queue_DequeueUpTo:1','fifo_queue_DequeueUpTo:2','fifo_queue_DequeueUpTo:3'])
        batch_size = length.shape[0]
        word = [id2word(w_id,self.vocab) for w_id in word_id]
        pos = [id2pos(p_id,self.pos_dict) for p_id in pos_id]

        total_sen = [Sentence([Token(n + 1, word[i][n], pos[i][n], None, None) for
                               n in range(length[i])]) for i in range(batch_size)]

        while True:
            batch_tree_word_id, batch_tree_pos_id, batch_buff_word_id, batch_buff_pos_id, batch_history_action_id, \
            batch_comp_head_order, batch_comp_dep_order, batch_comp_rel_id, batch_is_leaf, batch_stack_order, \
            batch_stack_length, batch_buff_length, batch_history_action_length = \
                [], [], [], [], [], [], [], [], [], [], [], [], []

            for sen in total_sen:
                if not sen.terminate:
                    tree_word_id, tree_pos_id, buff_word_id, buff_pos_id, history_action_id, comp_head_order, \
                    comp_dep_order, comp_rel_id, is_leaf, stack_order = self.parser.extract_from_current_state(sen)

                    batch_tree_word_id.append(tree_word_id)
                    batch_tree_pos_id.append(tree_pos_id)
                    batch_buff_word_id.append(buff_word_id)
                    batch_buff_pos_id.append(buff_pos_id)
                    batch_history_action_id.append(history_action_id)
                    batch_comp_head_order.append(comp_head_order)
                    batch_comp_dep_order.append(comp_dep_order)
                    batch_comp_rel_id.append(comp_rel_id)
                    batch_is_leaf.append(is_leaf)
                    batch_stack_order.append(stack_order)
                    batch_stack_length.append(len(stack_order))
                    batch_buff_length.append(len(buff_word_id))
                    batch_history_action_length.append(len(history_action_id))

            batch_tree_word_id = tf.keras.preprocessing.sequence.pad_sequences(batch_tree_word_id,
                                                                               dtype='int64', padding='post')
            batch_tree_pos_id = tf.keras.preprocessing.sequence.pad_sequences(batch_tree_pos_id,
                                                                              dtype='int64', padding='post')
            batch_buff_word_id = tf.keras.preprocessing.sequence.pad_sequences(batch_buff_word_id,
                                                                               dtype='int64', padding='post')
            batch_buff_pos_id = tf.keras.preprocessing.sequence.pad_sequences(batch_buff_pos_id,
                                                                              dtype='int64', padding='post')
            batch_history_action_id = tf.keras.preprocessing.sequence.pad_sequences(batch_history_action_id,
                                                                                    dtype='int64', padding='post')
            batch_comp_head_order = tf.keras.preprocessing.sequence.pad_sequences(batch_comp_head_order,
                                                                                  dtype='int64', padding='post')
            batch_comp_dep_order = tf.keras.preprocessing.sequence.pad_sequences(batch_comp_dep_order,
                                                                                 dtype='int64', padding='post')
            batch_comp_rel_id = tf.keras.preprocessing.sequence.pad_sequences(batch_comp_rel_id,
                                                                              dtype='int64', padding='post')
            batch_is_leaf = tf.keras.preprocessing.sequence.pad_sequences(batch_is_leaf,
                                                                          dtype='int64', padding='post')
            batch_stack_order = tf.keras.preprocessing.sequence.pad_sequences(batch_stack_order,
                                                                              dtype='int64', padding='post')

            feed_dict = {'tree_word_id:0': batch_tree_word_id, 'tree_pos_id:0': batch_tree_pos_id,
                         'buff_word_id:0': batch_buff_word_id, 'buff_pos_id:0': batch_buff_pos_id,
                         'history_action_id:0': batch_history_action_id, 'comp_head_order:0': batch_comp_head_order,
                         'comp_dep_order:0': batch_comp_dep_order, 'comp_rel_id:0': batch_comp_rel_id,
                         'is_leaf:0': batch_is_leaf, 'stack_order:0': batch_stack_order,
                         'stack_length:0': batch_stack_length, 'buff_length:0': batch_buff_length,
                         'history_action_length:0': batch_history_action_length}

            prob = run_context.session.run("Softmax:0", feed_dict)

            idx = 0
            for sen in total_sen:
                if not sen.terminate:
                    legal_transitions = self.parser.get_legal_transitions(sen)
                    transition = np.argmax(np.array(legal_transitions) * prob[idx])
                    if transition not in [0, 1]:
                        self.parser.update_composition(sen, transition)  # update composition
                    self.parser.update_state_by_transition(sen, transition)  # update stack and buff
                    idx += 1
                    if self.parser.terminal(sen):
                        sen.terminate = True
            if [sen.terminate for sen in total_sen] == [True] * batch_size:
                for i, sen in enumerate(total_sen):
                    self.pred_head.append([t.pred_head_id for t in sen.tokens])
                    self.pred_dep.append([t.pred_dep_id for t in sen.tokens])
                break

        pred_head = tf.keras.preprocessing.sequence.pad_sequences(self.pred_head,dtype='int64', padding='post')
        pred_dep = tf.keras.preprocessing.sequence.pad_sequences(self.pred_dep, dtype='int64', padding='post')
        return tf.train.SessionRunArgs(self.no_op,{'pred_head:0':pred_head,'pred_dep:0':pred_dep})