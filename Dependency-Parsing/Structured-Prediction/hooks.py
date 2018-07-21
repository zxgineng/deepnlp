import numpy as np
import tensorflow as tf

from utils import Config
from data_loader import ArcStandardParser, Token, Sentence, NULL, load_vocab, load_pos, load_dep


class BeamTrainHook(tf.train.SessionRunHook):
    def __init__(self, inputs, targets):
        self.idx = inputs['idx']
        self.word = inputs['word']
        self.pos = inputs['pos']
        self.length = inputs['length']
        self.action_seq = targets['action_seq']
        self.vocab = load_vocab()
        self.pos_dict = load_pos()
        self.dep_dict = load_dep()
        self.parser = ArcStandardParser()
        self.no_op = tf.no_op()

    def before_run(self, run_context):
        idx, word, pos, length, action_seq = run_context.session.run(
            [self.idx, self.word, self.pos, self.length, self.action_seq])
        batch_size = word.shape[0]
        total_sen = [Sentence(
            [Token(idx[i][n], word[i][n].decode(), pos[i][n].decode(), NULL, NULL) for n in range(length[i])]) for i in
            range(batch_size)]

        beam_path = [[total_sen[i]] for i in range(batch_size)]
        stop_id = []
        for a_step in range(2 * max(length) - 1):

            batch_word_id, batch_pos_id, batch_dep_id = [], [], []
            for i in range(batch_size):
                if i not in stop_id:
                    for sen in beam_path[i]:
                        if sen.bs_action_seq:
                            transition = sen.bs_action_seq[-1]
                            if transition != 0:
                                self.parser.update_child_dependencies(sen, transition)    # update left/right children
                            self.parser.update_state_by_transition(sen, transition)    # update stack and buff
                        word_id, pos_id, dep_id= self.parser.extract_for_current_state(sen, self.vocab, self.pos_dict, self.dep_dict)
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
                        sen.bs_next_legal_action = self.parser.get_legal_labels(sen)
                        for j, s in enumerate(batch_action_scores[score_idx]):
                            if sen.bs_next_legal_action[j] == 1:  # if action is legal
                                temp.append(sen.new_branch(
                                    s, j, (batch_word_id[score_idx], batch_pos_id[score_idx], batch_dep_id[score_idx])))
                        score_idx += 1

                    temp = sorted(temp, key=lambda x: -x.bs_score)
                    gold = [(n, sen) for n, sen in enumerate(temp) if sen.bs_action_seq == list(action_seq[i, :a_step + 1])]
                    assert len(gold) <= 1, 'find %d gold' % len(gold)

                    if gold[0][0] >= Config.model.beam_size:
                        # print('丢失gold',i)
                        # print([t.word for t in total_sen[i].tokens])
                        # print(action_seq[i])
                        # print(gold[0][1].bs_action_seq)
                        stop_id.append(i)
                        beam_path[i] = [gold[0][1]] + temp[:Config.model.beam_size - 1]
                    elif len(gold[0][1].bs_action_seq) == 2 * length[i] - 1:
                        # print('结束',i)
                        # print([t.word for t in total_sen[i].tokens])
                        # print(action_seq[i])
                        # print(gold[0][1].bs_action_seq)
                        stop_id.append(i)
                        temp.pop(gold[0][0])
                        beam_path[i] = [gold[0][1]] + temp[:Config.model.beam_size - 1]
                    else:
                        beam_path[i] = temp[:Config.model.beam_size]

            if sorted(stop_id) == list(range(batch_size)):
                break

        beam_search_word = np.zeros([Config.model.beam_size,1,Config.model.word_feature_num])
        beam_search_pos = np.zeros([Config.model.beam_size, 1, Config.model.pos_feature_num])
        beam_search_dep = np.zeros([Config.model.beam_size, 1, Config.model.dep_feature_num])
        beam_search_action = np.zeros([Config.model.beam_size,1])
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
            beam_search_word = np.concatenate([beam_search_word,np.array(word)],1)
            beam_search_pos = np.concatenate([beam_search_pos, np.array(pos)], 1)
            beam_search_dep = np.concatenate([beam_search_dep, np.array(dep)], 1)
            beam_search_action = np.concatenate([beam_search_action,action],1)

        beam_search_word = beam_search_word[:,1:,:]
        beam_search_pos = beam_search_pos[:,1:,:]
        beam_search_dep = beam_search_dep[:,1:,:]
        beam_search_action = beam_search_action[:,1:]

        return tf.train.SessionRunArgs(self.no_op,{'beam_search_word:0':beam_search_word,
                                                   'beam_search_pos:0':beam_search_pos,
                                                   'beam_search_dep:0':beam_search_dep,
                                                   'seg_id:0':seg_id,
                                                   'beam_search_action:0':beam_search_action})





