import numpy as np
import tensorflow as tf

from utils import Config
from data_loader import ArcStandardParser, Token, NULL, ROOT, load_vocab, load_pos, load_dep


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

    def before_run(self, run_context):
        idx, word, pos, length, action_seq = run_context.session.run(
            [self.idx, self.word, self.pos, self.length, self.action_seq])
        batch_size = word.shape[0]
        all_parser = [ArcStandardParser(
            [Token(idx[i][n], word[i][n].decode(), pos[i][n].decode(), NULL, NULL) for n in range(length[i])]) for i in
                      range(batch_size)]
        beam_path = [[]] * batch_size
        stop_id = []
        for a_step in range(2 * max(self.length) - 1):
            batch_word_id, batch_pos_id, batch_dep_id = [], [], []
            batch_legal_labels = []
            for i in range(batch_size):
                word_id, pos_id, dep_id = all_parser[i].extract_for_current_state(self.word, self.pos_dict,
                                                                                  self.dep_dict)
                legal_labels = all_parser[i].get_legal_labels()
                batch_legal_labels.append(legal_labels)

                batch_word_id.append(word_id)
                batch_pos_id.append(pos_id)
                batch_dep_id.append(dep_id)

            batch_action_scores = run_context.session.run('while/scores:0', {'while/word_feature_id:0': self.word_id,
                                                               'while/pos_feature_id:0': self.pos_id,
                                                               'while/dep_feature_id:0': self.dep_id})
            assert (len(batch_legal_labels),len(batch_legal_labels[0])) == batch_action_scores.shape

            for i in range(batch_size):
                if i not in stop_id:
                    if not beam_path[i]:
                        for j, s in enumerate(batch_action_scores[i]):
                            if batch_legal_labels[i][j] != 0:
                                beam_path[i].append([s, [j], [(batch_word_id[i], batch_pos_id[i], batch_dep_id[i])]])
                        beam_path[i] = sorted(beam_path[i], key=lambda x: -x[0])[:Config.model.beam_size]
                    else:
                        temp = []
                        for prev in beam_path[i]:
                            for j, s in enumerate(batch_action_scores[i]):
                                if batch_legal_labels[i][j] != 0:
                                    temp.append([prev[0] + s, prev[1] + [j],
                                                 prev[3] + [(batch_word_id[i], batch_pos_id[i], batch_dep_id[i])]])

                        temp = sorted(temp, key=lambda x: -x[0])
                        gold = [(n,path) for n,path in enumerate(temp) if path[1]==self.action_seq[i,:a_step + 1]]
                        assert len(gold) <= 1, 'find %d gold'% len(gold)

                        if gold[0][0] >= Config.model.beam_size:
                            stop_id.append(i)
                            beam_path[i] = [gold[0][1]] + temp[:Config.model.beam_size-1]
                        elif a_step == 2*length[i] - 1:
                            stop_id.append(i)
                            temp.pop(gold[0][0])
                            beam_path[i] = [gold[0][1]] + temp[:Config.model.beam_size-1]
            if stop_id == list(range(batch_size)):
                break





