import tensorflow as tf
import os

from utils import Config


class PRFScoreHook(tf.train.SessionRunHook):
    def __init__(self, model_dir=None):
        self.TP_num = 0
        self.pred_num = 0
        self.gold_num = 0
        if model_dir:
            self.model_dir = os.path.join(Config.train.model_dir, 'eval_' + model_dir)
        else:
            self.model_dir = os.path.join(Config.train.model_dir, 'eval')

    def begin(self):
        self._summary_writer = tf.summary.FileWriter(self.model_dir)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(['val/IteratorGetNext:1', 'val/IteratorGetNext:2', 'prediction:0'])

    def after_run(self, run_context, run_values):
        labels, length, preds = run_values.results
        for i in range(len(labels)):
            la = labels[i]
            le = length[i]
            pr = preds[i]

            gold_seg = []
            pred_seg = []
            la_start = 5
            pr_start = 5

            for n in range(6, le - 6):

                if la[n] == 0 or la[n] == 3:
                    gold_seg.append((la_start, n))
                    la_start = n

                if pr[n] == 0 or pr[n] == 3:
                    pred_seg.append((pr_start, n))
                    pr_start = n

            gold_seg.append((la_start, le))

            pred_seg.append((pr_start, le))

            self.gold_num += len(gold_seg)
            self.pred_num += len(pred_seg)
            TP = set(gold_seg) & set(pred_seg)
            self.TP_num += len(TP)

    def end(self, session):

        global_step = session.run(tf.train.get_global_step())
        summary_op = tf.get_default_graph().get_collection('prf')

        p = self.TP_num / self.pred_num
        r = self.TP_num / self.gold_num
        f1 = 2 * p * r / (p + r)

        summary = session.run(summary_op, {'p_ph:0': p,
                                           'r_ph:0': r,
                                           'f1_ph:0': f1})
        for s in summary:
            self._summary_writer.add_summary(s, global_step)

        self._summary_writer.flush()

        print('*' * 40)
        print("epoch", Config.train.epoch + 1, 'finished')
        print('precision:', p, '    recall:', r, '    f1:', f1)
        print('*' * 40)
