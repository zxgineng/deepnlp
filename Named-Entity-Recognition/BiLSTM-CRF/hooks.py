import tensorflow as tf
import os

from utils import Config
import data_loader


class PRFScoreHook(tf.train.SessionRunHook):
    def __init__(self, model_dir=None):
        self.tp_num = 0
        self.pred_num = 0
        self.gold_num = 0
        if model_dir:
            self.model_dir = os.path.join(Config.train.model_dir, 'eval_' + model_dir)
        else:
            self.model_dir = os.path.join(Config.train.model_dir, 'eval')

        self.tag = data_loader.load_tag()

    def begin(self):
        self._summary_writer = tf.summary.FileWriter(self.model_dir)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(['val/IteratorGetNext:1', 'val/IteratorGetNext:2', 'prediction:0'])

    def after_run(self, run_context, run_values):
        labels, length, preds = run_values.results
        for i in range(len(labels)):
            la = data_loader.id2label(labels[i], self.tag)
            le = length[i]
            pr = data_loader.id2label(preds[i], self.tag)

            gold_seg = []
            pred_seg = []
            la_start = 0
            pr_start = 0

            la_t = la[0][2:] if la[0][0] in ['B', 'S'] else None
            pr_t = pr[0][2:] if pr[0][0] in ['B', 'S'] else None

            for n in range(1, le):

                if la[n][0] == 'B' or la[n][0] == 'S':
                    if la_t:
                        gold_seg.append((la_start, n, la_t))
                    la_start = n
                    la_t = la[n][2:]

                elif la[n][0] == 'O':
                    if la_t:
                        gold_seg.append((la_start, n, la_t))
                    la_t = None

                if pr[n][0] == 'B' or pr[n][0] == 'S':
                    if pr_t:
                        pred_seg.append((pr_start, n, pr_t))
                    pr_start = n
                    pr_t = pr[n][2:]

                elif pr[n][0] == 'O':
                    if pr_t:
                        pred_seg.append((pr_start, n, pr_t))
                    pr_t = None

            if la_t:
                gold_seg.append((la_start, le, la_t))
            if pr_t:
                pred_seg.append((pr_start, le, pr_t))

            self.gold_num += len(gold_seg)
            self.pred_num += len(pred_seg)
            tp = set(gold_seg) & set(pred_seg)
            self.tp_num += len(tp)

    def end(self, session):

        global_step = session.run(tf.train.get_global_step())
        summary_op = tf.get_default_graph().get_collection('prf')

        p = self.tp_num / self.pred_num
        r = self.tp_num / self.gold_num
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
