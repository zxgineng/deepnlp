import numpy as np
import tensorflow as tf

from utils import Config


class BeamSearchHook(tf.train.SessionRunHook):
    def __init__(self, inputs, targets):
        self.word_id = inputs['word_id']
        self.pos_id = inputs['pos_id']
        self.length = inputs['length']
        self.seq_labels = targets['seq_labels']

    def before_run(self, run_context):
        total_scores = run_context.sess.run('scores:0', {'word_id:0': self.word_id, 'pos_id:0': self.pos_id,
                                                         'dep_id:0': self.dep_id})
        batch_size = total_scores.shape[0]

        beam_path = [[]] * batch_size

        early_stop_id = []  # todo 用一个和length样式一样的转为mask
        for a in range(max(2 * max(self.length) - 1)):
            for i in range(batch_size):
                if i in early_stop_id:
                    pass
                    # todo 加0？

                if not beam_path[i]:
                    for j, s in enumerate(total_scores[i]):
                        beam_path[i].append([s, [j], [(self.word_id[i], self.pos_id[i], self.dep_id[i])]])
                    beam_path[i] = sorted(beam_path[i], key=lambda x: -x[0])[:Config.model.beam_size]
                else:
                    temp = []
                    for prev in beam_path[i]:
                        for j, s in enumerate(total_scores[i]):
                            if self.seq_labels[i][:a + 1] == prev[1] + [j]:
                                gold_inputs_seq = prev[3] + [(self.word_id[i], self.pos_id[i], self.dep_id[i])]
                            temp.append([prev[0] + s, prev[1] + [j],
                                         prev[3] + [(self.word_id[i], self.pos_id[i], self.dep_id[i])]])

                    beam_path[i] = sorted(temp, key=lambda x: -x[0])[:Config.model.beam_size]

                    if self.seq_labels[i][:a + 1][:self.length[i]] not in [p[1] for p in beam_path[i]]:
                        early_stop_id.append(i)
                        beam_path[i] = [[0, self.seq_labels[i][:a + 1], gold_inputs_seq]] + beam_path[i][1:]

                    # todo 之后怎么并行运算?

                # transition
