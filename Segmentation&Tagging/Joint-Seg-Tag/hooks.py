import tensorflow as tf
import os

from utils import Config
import data_loader


class PRFScoreHook(tf.train.SessionRunHook):
    def __init__(self, model_dir=None):
        self.seg_tp_num = 0
        self.tag_tp_num = 0
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

            gold_tag = []
            pred_tag = []
            la_t = la[0][2:]
            pr_t = pr[0][2:]

            for n in range(1, le):

                if la[n][0] == 'B' or la[n][0] == 'S':
                    gold_seg.append((la_start, n))
                    gold_tag.append((la_start, n, la_t))
                    la_start = n
                    la_t = la[n][2:]

                if pr[n][0] == 'B' or pr[n][0] == 'S':
                    pred_seg.append((pr_start, n))
                    pred_tag.append((pr_start, n, pr_t))
                    pr_start = n
                    pr_t = pr[n][2:]

            gold_seg.append((la_start, le))

            pred_seg.append((pr_start, le))

            gold_tag.append((la_start, le, la_t))
            pred_tag.append((pr_start, le, pr_t))

            self.gold_num += len(gold_seg)
            self.pred_num += len(pred_seg)
            seg_tp = set(gold_seg) & set(pred_seg)
            tag_tp = set(gold_tag) & set(pred_tag)
            self.seg_tp_num += len(seg_tp)
            self.tag_tp_num += len(tag_tp)

    def end(self, session):

        global_step = session.run(tf.train.get_global_step())
        summary_op = tf.get_default_graph().get_collection('prf')

        seg_p = self.seg_tp_num / self.pred_num
        seg_r = self.seg_tp_num / self.gold_num
        seg_f1 = 2 * seg_p * seg_r / (seg_p + seg_r)

        tag_p = self.tag_tp_num / self.pred_num
        tag_r = self.tag_tp_num / self.gold_num
        tag_f1 = 2 * tag_p * tag_r / (tag_p + tag_r)

        summary = session.run(summary_op, {'seg_p_ph:0': seg_p,
                                           'seg_r_ph:0': seg_r,
                                           'seg_f1_ph:0': seg_f1,
                                           'tag_p_ph:0': tag_p,
                                           'tag_r_ph:0': tag_r,
                                           'tag_f1_ph:0': tag_f1})
        for s in summary:
            self._summary_writer.add_summary(s, global_step)

        self._summary_writer.flush()

        print('*' * 40)
        print("epoch", Config.train.epoch + 1, 'finished')
        print('seg-precision:', seg_p, '    seg-recall:', seg_r, '    seg-f1:', seg_f1)
        print('tag-precision:', tag_p, '    tag-recall:', tag_r, '    tag-f1:', tag_f1)
        print('*' * 40)
