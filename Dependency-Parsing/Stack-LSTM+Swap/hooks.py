import numpy as np
import tensorflow as tf
import os

from utils import Config
from data_loader import ArcStandardParser, Token, Sentence, dep2id, load_dep


class BeamSearchHook(tf.train.SessionRunHook):
    def __init__(self, inputs, targets, mode='train'):
        self.idx = inputs['idx']
        self.word = inputs['word']
        self.pos = inputs['pos']
        self.length = inputs['length']
        self.action_seq = targets['action_seq']
        self.head_id = targets['arc']
        self.dep_id = targets['dep_id']
        self.parser = ArcStandardParser()
        self.no_op = tf.no_op()
        self.dep_dict = load_dep()

        self.mode = mode
        if mode == 'eval':
            self.total = 0
            self.correct_head = 0
            self.correct_dep = 0
            self.model_dir = os.path.join(Config.train.model_dir, 'eval')

    def before_run(self, run_context):
        idx, word, pos, length, action_seq, head_id, gold_dep_id = run_context.session.run(
            [self.idx, self.word, self.pos, self.length, self.action_seq, self.head_id, self.dep_id])
        batch_size = word.shape[0]
        total_sen = [Sentence(
            [Token(idx[i][n], word[i][n].decode(), pos[i][n].decode(), None, None) for n in range(length[i])]) for i in
            range(batch_size)]

        fall_out_count = 0
        end_count = 0

        beam_path = [[total_sen[i]] for i in range(batch_size)]
        stop_id = []
        for a_step in range(2 * max(length)):
            batch_word_id, batch_pos_id, batch_dep_id = [], [], []
            for i in range(batch_size):
                if i not in stop_id:
                    for sen in beam_path[i]:
                        if sen.bs_action_seq:
                            transition = sen.bs_action_seq[-1]
                            if transition != 0:
                                self.parser.update_child_dependencies(sen, transition)  # update left/right children
                            self.parser.update_state_by_transition(sen, transition)  # update stack and buff
                        word_id, pos_id, dep_id = self.parser.extract_for_current_state(sen)
                        batch_word_id.append(word_id)
                        batch_pos_id.append(pos_id)
                        batch_dep_id.append(dep_id)
            batch_action_scores = run_context.session.run('scores:0', {'word_feature_id:0': batch_word_id,
                                                                       'pos_feature_id:0': batch_pos_id,
                                                                       'dep_feature_id:0': batch_dep_id})

            score_idx = 0
            for i in range(batch_size):
                if i not in stop_id:
                    temp = []
                    for sen in beam_path[i]:
                        sen.bs_legal_transitions = self.parser.get_legal_transitions(sen)
                        for j, s in enumerate(batch_action_scores[score_idx]):
                            if sen.bs_legal_transitions[j] == 1:
                                temp.append(sen.new_branch(
                                    s, j, (batch_word_id[score_idx], batch_pos_id[score_idx], batch_dep_id[score_idx])))
                        score_idx += 1
                    temp = sorted(temp, key=lambda x: -x.bs_score)
                    gold = [(n, sen) for n, sen in enumerate(temp) if
                            sen.bs_action_seq == list(action_seq[i, :a_step + 1])]
                    assert len(gold) == 1, 'find %d gold' % len(gold)

                    if gold[0][0] >= Config.model.beam_size:
                        # cal eval score
                        if self.mode == 'eval':
                            best_sen = temp[0]
                            transition = best_sen.bs_action_seq[-1]
                            if transition != 0:
                                self.parser.update_child_dependencies(best_sen,
                                                                      transition)  # update left/right children
                            self.parser.update_state_by_transition(best_sen, transition)  # update stack and buff
                            gold_head = head_id[i, :length[i]]
                            pred_head = np.array([t.head_id for t in best_sen.tokens])
                            assert len(gold_head) == len(pred_head)
                            self.total += len(gold_head)
                            correct_head = gold_head == pred_head
                            self.correct_head += np.sum(correct_head)
                            gold_dep = gold_dep_id[i, :length[i]]
                            pred_dep = np.array(dep2id([t.dep for t in best_sen.tokens], self.dep_dict))
                            correct_dep = gold_dep == pred_dep
                            self.correct_dep += np.sum(np.logical_and(correct_head, correct_dep))
                        fall_out_count += 1
                        stop_id.append(i)
                        beam_path[i] = [gold[0][1]] + temp[:Config.model.beam_size - 1]
                    elif len(gold[0][1].bs_action_seq) == 2 * length[i]:
                        # cal eval score
                        if self.mode == 'eval':
                            best_sen = temp[0]
                            transition = best_sen.bs_action_seq[-1]
                            if transition != 0:
                                self.parser.update_child_dependencies(best_sen,
                                                                      transition)  # update left/right children
                            self.parser.update_state_by_transition(best_sen, transition)  # update stack and buff
                            gold_head = head_id[i, :length[i]]
                            pred_head = np.array([t.head_id for t in best_sen.tokens])
                            assert len(gold_head) == len(pred_head)
                            self.total += len(gold_head)
                            correct_head = gold_head == pred_head
                            self.correct_head += np.sum(correct_head)
                            gold_dep = gold_dep_id[i, :length[i]]
                            pred_dep = np.array(dep2id([t.dep for t in best_sen.tokens], self.dep_dict))
                            correct_dep = gold_dep == pred_dep
                            self.correct_dep += np.sum(np.logical_and(correct_head, correct_dep))

                        end_count += 1
                        stop_id.append(i)
                        temp.pop(gold[0][0])
                        beam_path[i] = [gold[0][1]] + temp[:Config.model.beam_size - 1]
                    else:
                        beam_path[i] = temp[:Config.model.beam_size]

            if sorted(stop_id) == list(range(batch_size)):
                print('fall out count:',fall_out_count)
                print('end count:',end_count)
                print('step:',a_step)
                print('*'*40)
                break

        beam_search_word = np.zeros([Config.model.beam_size, 1, Config.model.word_feature_num])
        beam_search_pos = np.zeros([Config.model.beam_size, 1, Config.model.pos_feature_num])
        beam_search_dep = np.zeros([Config.model.beam_size, 1, Config.model.dep_feature_num])
        beam_search_action = np.zeros([Config.model.beam_size, 1])
        seg_id = [0]
        for i in range(batch_size):
            word = []
            pos = []
            dep = []
            action = []
            for sen in beam_path[i]:
                word_seq = []
                pos_seq = []
                dep_seq = []
                for input in sen.bs_input_seq:
                    word_seq.append(input[0])
                    pos_seq.append(input[1])
                    dep_seq.append(input[2])
                word.append(word_seq)
                pos.append(pos_seq)
                dep.append(dep_seq)
                action.append(sen.bs_action_seq)
            seg_id.append(seg_id[-1] + len(word[0]))
            beam_search_word = np.concatenate([beam_search_word, np.array(word)], 1)
            beam_search_pos = np.concatenate([beam_search_pos, np.array(pos)], 1)
            beam_search_dep = np.concatenate([beam_search_dep, np.array(dep)], 1)
            beam_search_action = np.concatenate([beam_search_action, action], 1)

        beam_search_word = beam_search_word[:, 1:, :]
        beam_search_pos = beam_search_pos[:, 1:, :]
        beam_search_dep = beam_search_dep[:, 1:, :]
        beam_search_action = beam_search_action[:, 1:]

        return tf.train.SessionRunArgs(self.no_op, {'beam_search_word:0': beam_search_word,
                                                    'beam_search_pos:0': beam_search_pos,
                                                    'beam_search_dep:0': beam_search_dep,
                                                    'seg_id:0': seg_id,
                                                    'beam_search_action:0': beam_search_action})

    def end(self, session):
        if self.mode == 'eval':
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

