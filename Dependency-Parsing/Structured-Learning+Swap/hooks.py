import numpy as np
import tensorflow as tf
import multiprocessing

from utils import Config
from data_loader import ArcStandardParser, Token, Sentence


# def _select_branch(sen, temp, batch_word_id, batch_pos_id, batch_dep_id, batch_action_scores,score_idx,parser):
#     if sen.terminate:
#         sen.bs_input_seq.append(([-1] * Config.model.word_feature_num,
#                                  [-1] * Config.model.pos_feature_num,
#                                  [-1] * Config.model.dp_feature_num))
#         sen.bs_transition_seq.append(-1)
#         temp.append(sen)
#     else:
#         sen.bs_legal_transition = parser.get_legal_transitions(sen)
#         for j, s in enumerate(batch_action_scores[score_idx]):
#             if sen.bs_legal_transition[j] == 1:
#                 temp.append(sen.new_branch(
#                     s, j, (batch_word_id[score_idx], batch_pos_id[score_idx], batch_dep_id[score_idx])))


class BeamTrainHook(tf.train.SessionRunHook):
    def __init__(self, inputs, targets):
        self.idx = inputs['idx']
        self.word = inputs['word']
        self.pos = inputs['pos']
        self.sen_length = inputs['sen_length']
        self.transition_seq = targets['transition_seq']
        self.tran_length = targets['tran_length']
        self.parser = ArcStandardParser()
        self.no_op = tf.no_op()

    def before_run(self, run_context):
        idx, word, pos, sen_length, transition_seq,tran_length = run_context.session.run(
            [self.idx, self.word, self.pos, self.sen_length, self.transition_seq, self.tran_length])
        batch_size = word.shape[0]
        total_sen = [Sentence([Token(idx[i][n], word[i][n].decode(), pos[i][n].decode(), None,None) for
                               n in range(sen_length[i])]) for i in range(batch_size)]

        all_bs_seq = [[total_sen[i]] for i in range(batch_size)]  # store all beam search branch
        stopped_bs = [False] * batch_size  # store whether beam search decode of a sentence is over or not
        bs_tran_length = [[] for _ in range(batch_size)]  # store transition length of all beam search branch

        t_step = 0
        fall_out_count = 0
        end_count = 0
        while True:
            # extract features from all beam search candidates and calculate scores of next transitions
            batch_word_id, batch_pos_id, batch_dep_id = [], [], []
            for i in range(batch_size):
                if not stopped_bs[i]:
                    for sen in all_bs_seq[i]:
                        if not sen.terminate:
                            word_id, pos_id, dep_id= self.parser.extract_for_current_state(sen)
                            batch_word_id.append(word_id)
                            batch_pos_id.append(pos_id)
                            batch_dep_id.append(dep_id)
            batch_action_scores = run_context.session.run('scores:0', {'word_feature_id:0': batch_word_id,
                                                                       'pos_feature_id:0': batch_pos_id,
                                                                       'dep_feature_id:0': batch_dep_id})

            score_idx = 0
            for i in range(batch_size):
                if not stopped_bs[i]:
                    temp = []
                    # temp = multiprocessing.Manager().list()
                    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                    for sen in all_bs_seq[i]:
                        # pool.apply_async(_select_branch, args=(s, j, sen, temp, batch_word_id, batch_pos_id, batch_dep_id, score_idx))
                        # pool.close()
                        # pool.join()

                        if sen.terminate:
                            # keep the seq length of all beam search paths of one sentence same
                            sen.bs_input_seq.append(([0]*Config.model.word_feature_num,
                                                     [0]*Config.model.pos_feature_num,
                                                     [0]*Config.model.dep_feature_num))
                            sen.bs_transition_seq.append(0)
                            temp.append(sen)
                        else:
                            sen.bs_legal_transition = self.parser.get_legal_transitions(sen)
                            for j, s in enumerate(batch_action_scores[score_idx]):
                                if sen.bs_legal_transition[j] == 1:
                                    # store all possible transitions of current branch
                                    temp.append(sen.new_branch(
                                        s, j, (batch_word_id[score_idx], batch_pos_id[score_idx], batch_dep_id[score_idx])))
                            score_idx += 1

                    temp = sorted(temp, key=lambda x: -x.bs_ave_score)
                    gold = [(n, sen) for n, sen in enumerate(temp) if sen.bs_transition_seq == list(transition_seq[i, :t_step + 1])]
                    assert len(gold) == 1, 'find %d gold' % len(gold)

                    if gold[0][0] >= Config.model.beam_size:    # gold fall out
                        stopped_bs[i] = True
                        fall_out_count+=1
                        all_bs_seq[i] = [gold[0][1]] + temp[:Config.model.beam_size - 1]
                        bs_tran_length[i] = [sen.bs_tran_length for sen in all_bs_seq[i]]
                    else:
                        all_bs_seq[i] = temp[:Config.model.beam_size]
                        stopped_list = []
                        for j,sen in enumerate(all_bs_seq[i]):
                            if not sen.terminate:
                                transition = sen.bs_transition_seq[-1]
                                if transition not in [0, 1]:
                                    self.parser.update_child_dependencies(sen, transition)  # update left/right children
                                self.parser.update_state_by_transition(sen, transition)  # update stack and buff
                                if self.parser.terminal(sen):
                                    sen.terminate = True
                            stopped_list.append(sen.terminate)
                        if stopped_list == [True] * len(all_bs_seq[i]):
                            stopped_bs[i] = True
                            end_count+=1
                            all_bs_seq[i].pop(gold[0][0])
                            all_bs_seq[i] = [gold[0][1]] + all_bs_seq[i]
                            bs_tran_length[i] = [sen.bs_tran_length for sen in all_bs_seq[i]]

            if stopped_bs == [True] * batch_size:
                print('step:', t_step)
                print('fall_out:',fall_out_count)
                print('end:',end_count)
                print('*'*40)
                break

            t_step += 1

        beam_search_word = np.zeros([Config.model.beam_size,1,Config.model.word_feature_num])
        beam_search_pos = np.zeros([Config.model.beam_size, 1, Config.model.pos_feature_num])
        beam_search_dep = np.zeros([Config.model.beam_size, 1, Config.model.dep_feature_num])
        beam_search_action = np.zeros([Config.model.beam_size,1])
        slicer = [0]
        for i in range(batch_size):
            word = []
            pos = []
            dep = []
            action = []
            for sen in all_bs_seq[i]:
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
                action.append(sen.bs_transition_seq)
            slicer.append(slicer[-1]+len(word[0]))
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
                                                   'slicer:0':slicer,
                                                   'bs_tran_length:0':bs_tran_length,
                                                   'beam_search_action:0':beam_search_action})





