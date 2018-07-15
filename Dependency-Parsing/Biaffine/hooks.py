import numpy as np
from collections import defaultdict
import tensorflow as tf
import os

from utils import Config


def mst(scores):
    """
    Chu-Liu-Edmonds' algorithm for finding minimum spanning arborescence in graphs.
    Calculates the arborescence with node 0 as root.
    Source: https://github.com/chantera/biaffineparser/blob/master/utils.py

    :param scores: `scores[i][j]` is the weight of edge from node `i` to node `j`
    :returns an array containing the head node (node with edge pointing to current node) for each node,
             with head[0] fixed as 0
    """

    scores = scores - np.max(scores, -1, keepdims=True)
    scores = np.exp(scores) / np.sum(np.exp(scores), -1, keepdims=True)

    length = scores.shape[0]
    scores = scores * (1 - np.eye(length))
    heads = np.argmax(scores, axis=1)
    heads[0] = 0
    tokens = np.arange(1, length)
    roots = np.where(heads[tokens] == 0)[0] + 1

    if len(roots) < 1:
        root_scores = scores[tokens, 0]
        head_scores = scores[tokens, heads[tokens]]
        new_root = tokens[np.argmax(root_scores / head_scores)]
        heads[new_root] = 0
    elif len(roots) > 1:
        root_scores = scores[roots, 0]
        scores[roots, 0] = 0
        new_heads = np.argmax(scores[roots][:, tokens], axis=1) + 1
        new_root = roots[np.argmin(
            scores[roots, new_heads] / root_scores)]
        heads[roots] = new_heads
        heads[new_root] = 0

    edges = defaultdict(set)
    vertices = set((0,))
    for dep, head in enumerate(heads[tokens]):
        vertices.add(dep + 1)
        edges[head].add(dep + 1)
    for cycle in _find_cycle(vertices, edges):
        dependents = set()
        to_visit = set(cycle)
        while len(to_visit) > 0:
            node = to_visit.pop()
            if node not in dependents:
                dependents.add(node)
                to_visit.update(edges[node])
        cycle = np.array(list(cycle))
        old_heads = heads[cycle]
        old_scores = scores[cycle, old_heads]
        non_heads = np.array(list(dependents))
        scores[np.repeat(cycle, len(non_heads)),
               np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
        new_heads = np.argmax(scores[cycle][:, tokens], axis=1) + 1
        new_scores = scores[cycle, new_heads] / old_scores
        change = np.argmax(new_scores)
        changed_cycle = cycle[change]
        old_head = old_heads[change]
        new_head = new_heads[change]
        heads[changed_cycle] = new_head
        edges[new_head].add(changed_cycle)
        edges[old_head].remove(changed_cycle)

    return heads


def _find_cycle(vertices, edges):
    """
    https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/etc/tarjan.py
    """
    _index = [0]
    _stack = []
    _indices = {}
    _lowlinks = {}
    _onstack = defaultdict(lambda: False)
    _SCCs = []

    def _strongconnect(v):
        _indices[v] = _index[0]
        _lowlinks[v] = _index[0]
        _index[0] += 1
        _stack.append(v)
        _onstack[v] = True

        for w in edges[v]:
            if w not in _indices:
                _strongconnect(w)
                _lowlinks[v] = min(_lowlinks[v], _lowlinks[w])
            elif _onstack[w]:
                _lowlinks[v] = min(_lowlinks[v], _indices[w])

        if _lowlinks[v] == _indices[v]:
            SCC = set()
            while True:
                w = _stack.pop()
                _onstack[w] = False
                SCC.add(w)
                if not (w != v):
                    break
            _SCCs.append(SCC)

    for v in vertices:
        if v not in _indices:
            _strongconnect(v)

    return [SCC for SCC in _SCCs if len(SCC) > 1]


class MSTHook(tf.train.SessionRunHook):
    def __init__(self, model_dir=None):
        self.total = 0
        self.correct_arc = 0
        self.correct_label = 0
        if model_dir:
            self.model_dir = os.path.join(Config.train.model_dir, 'eval_' + model_dir)
        else:
            self.model_dir = os.path.join(Config.train.model_dir, 'eval')

    def begin(self):
        self._summary_writer = tf.summary.FileWriter(self.model_dir)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            ["arc_logits:0", "label_logits:0", "val/IteratorGetNext:2", "val/IteratorGetNext:3",
             "val/IteratorGetNext:4"])

    def after_run(self, run_context, run_values):
        arc_logits, label_logits, arc, label, length = run_values.results
        for i in range(len(length)):
            arc_gold = arc[i, :length[i] - 1]
            label_gold = label[i, :length[i] - 1]
            arc_pred = mst(arc_logits[i, :length[i], :length[i]])[1:]
            correct_arc = arc_pred == arc_gold
            self.correct_arc += np.sum(correct_arc)
            self.total += len(arc_gold)
            label_score = label_logits[i, range(1, length[i]), arc_pred, :]
            label_pred = np.argmax(label_score, -1)
            correct_label = label_pred == label_gold
            self.correct_label += np.sum(np.logical_and(correct_arc, correct_label))

    def end(self, session):

        global_step = session.run(tf.train.get_global_step())
        summary_op = tf.get_default_graph().get_collection('acc')

        arc_acc = self.correct_arc / self.total
        label_acc = self.correct_label / self.total

        summary = session.run(summary_op, {'arc_ph:0': arc_acc,
                                           'label_ph:0': label_acc})
        for s in summary:
            self._summary_writer.add_summary(s, global_step)

        self._summary_writer.flush()

        print('*' * 40)
        print("epoch", Config.train.epoch + 1, 'finished')
        print('UAS:', arc_acc, '    LAS:', label_acc)
        print('*' * 40)