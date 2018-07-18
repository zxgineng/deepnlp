import numpy as np
import tensorflow as tf

from utils import Config


class BeamSearchHook(tf.train.SessionRunHook):
    def __init__(self, inputs, targets):
        self.word_id = inputs['word_id']
        self.pos_id = inputs['pos_id']
        self.dep_id = inputs['dep_id']
        self.seq_labels = targets['seq_labels']
        self.length = targets['length']

    def before_run(self, run_context):
        total_scores = run_context.sess.run('scores:0', {'word_id:0': self.word_id, 'pos_id:0': self.pos_id,
                                                         'dep_id:0': self.dep_id})  # batch * 79
        batch_size = total_scores.shape[0]

        beam_path = [[[0.0]] * Config.model.beam_size] * batch_size

        for a in range(max(2 * self.length - 1)):
            for i in range(batch_size):
                for j, s in enumerate(total_scores[i]):
                    if len(beam_path[i][j]) == 1:
                        beam_path[i][j][0] += s
                        beam_path[i][j].append((self.word_id[i], self.pos_id[i], self.dep_id[i]))
                        # transition
