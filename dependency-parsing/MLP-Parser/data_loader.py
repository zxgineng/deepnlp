import argparse
import os
import sys
import numpy as np
import tensorflow as tf

from utils import Config


NULL = "<null>"
UNK = "<unk>"
ROOT = "<root>"
pos_prefix = "<p>:"
dep_prefix = "<d>:"
punc_pos = ["''", "``", ":", ".", ","]


class Token():
    def __init__(self, token_id, word, pos, dep, head_id):
        self.token_id = token_id
        self.word = word.lower()
        self.pos = pos_prefix + pos
        self.dep = dep_prefix + dep
        self.head_id = head_id
        self.predicted_head_id = None
        self.left_children = list()
        self.right_children = list()


    def is_root_token(self):
        if self.word == ROOT:
            return True
        return False


    def is_null_token(self):
        if self.word == NULL:
            return True
        return False


    def is_unk_token(self):
        if self.word == UNK:
            return True
        return False


    def reset_predicted_head_id(self):
        self.predicted_head_id = None

NULL_TOKEN = Token(0, NULL, NULL, NULL, -1)
ROOT_TOKEN = Token(0, ROOT, ROOT, ROOT, -1)
UNK_TOKEN = Token(0, UNK, UNK, UNK, -1)

class Sentence():
    def __init__(self, tokens):
        self.Root = Token(0, ROOT, ROOT, ROOT, -1)
        self.tokens = tokens
        self.buff = [token for token in self.tokens]
        self.stack = [self.Root]
        self.dependencies = []
        self.predicted_dependencies = []


    def load_gold_dependency_mapping(self):
        for token in self.tokens:
            if token.head_id != -1:
                token.parent = self.tokens[token.head_id]
                if token.head_id > token.token_id:
                    token.parent.left_children.append(token.token_id)
                else:
                    token.parent.right_children.append(token.token_id)
            else:
                token.parent = self.Root

        for token in self.tokens:
            token.left_children.sort()
            token.right_children.sort()


    def update_child_dependencies(self, curr_transition):
        if curr_transition == 0:
            head = self.stack[-1]
            dependent = self.stack[-2]
        elif curr_transition == 1:
            head = self.stack[-2]
            dependent = self.stack[-1]

        if head.token_id > dependent.token_id:
            head.left_children.append(dependent.token_id)
            head.left_children.sort()
        else:
            head.right_children.append(dependent.token_id)
            #按照id顺序排序
            head.right_children.sort()
            # dependent.head_id = head.token_id


    def get_child_by_index_and_depth(self, token, index, direction, depth):  # Get child token
        if depth == 0:
            return token

        if direction == "left":
            if len(token.left_children) > index:
                return self.get_child_by_index_and_depth(
                    self.tokens[token.left_children[index]], index, direction, depth - 1)
            return NULL_TOKEN
        else:
            if len(token.right_children) > index:
                return self.get_child_by_index_and_depth(
                    self.tokens[token.right_children[::-1][index]], index, direction, depth - 1)
            return NULL_TOKEN


    def get_legal_labels(self):
        labels = ([1] if len(self.stack) > 2 else [0])
        labels += ([1] if len(self.stack) >= 2 else [0])
        labels += [1] if len(self.buff) > 0 else [0]
        return labels


    def get_transition_from_current_state(self):  # logic to get next transition
        if len(self.stack) < 2:
            return 2  # shift

        stack_token_0 = self.stack[-1]
        stack_token_1 = self.stack[-2]
        if stack_token_1.token_id >= 0 and stack_token_1.head_id == stack_token_0.token_id:  # left arc
            return 0
        elif stack_token_1.token_id >= -1 and stack_token_0.head_id == stack_token_1.token_id \
                and stack_token_0.token_id not in map(lambda x: x.head_id, self.buff):
            return 1  # right arc
        else:
            return 2 if len(self.buff) != 0 else None


    def update_state_by_transition(self, transition, gold=True):  # updates stack, buffer and dependencies
        if transition is not None:
            if transition == 2:  # shift
                self.stack.append(self.buff[0])
                self.buff = self.buff[1:] if len(self.buff) > 1 else []
            elif transition == 0:  # left arc
                #记录
                self.dependencies.append(
                    (self.stack[-1], self.stack[-2])) if gold else self.predicted_dependencies.append(
                    (self.stack[-1], self.stack[-2]))
                #合并
                self.stack = self.stack[:-2] + self.stack[-1:]
            elif transition == 1:  # right arc
                self.dependencies.append(
                    (self.stack[-2], self.stack[-1])) if gold else self.predicted_dependencies.append(
                    (self.stack[-2], self.stack[-1]))
                self.stack = self.stack[:-1]


    def reset_to_initial_state(self):
        self.buff = [token for token in self.tokens]
        self.stack = [self.Root]


    def clear_prediction_dependencies(self):
        self.predicted_dependencies = []


    def clear_children_info(self):
        for token in self.tokens:
            token.left_children = []
            token.right_children = []


def build_vocab():
    train_data = read_file('train.conll')
    val_data = read_file('dev.conll')
    test_data = read_file('test.conll')

    for sentence in train_data + val_data:
        all_words = set([token.word for token in sentence.tokens])
        all_pos = set([token.pos for token in sentence.tokens])
        all_dep = set([token.dep for token in sentence.tokens])

    all_words.add(ROOT_TOKEN.word)
    all_words.add(NULL_TOKEN.word)
    all_words.add(UNK_TOKEN.word)

    all_pos.add(ROOT_TOKEN.pos)
    all_pos.add(NULL_TOKEN.pos)
    all_pos.add(UNK_TOKEN.pos)

    all_dep.add(ROOT_TOKEN.dep)
    all_dep.add(NULL_TOKEN.dep)
    all_dep.add(UNK_TOKEN.dep)

    word_vocab = list(all_words)
    pos_vocab = list(all_pos)
    dep_vocab = list(all_dep)

    #.....



def convert_sentence(token_lines):
    """convert tokens to a sentence object"""
    tokens = []
    for line in token_lines:
        fields = line.strip().split("\t")
        token_index = int(fields[0])
        word = fields[1]
        pos = fields[4]
        dep = fields[7]
        head_index = int(fields[6])
        token = Token(token_index, word, pos, dep, head_index)
        tokens.append(token)
    sentence = Sentence(tokens)
    return sentence


def read_file(file):
    """read file and return a list of sentence object"""
    path = os.path.join(Config.data.base_path,file)
    all_sentences = []
    token_lines = []
    with open(path,encoding='utf8') as f:
        for line in f:
            token_raw = line.strip()
            if len(token_raw) > 0:
                token_lines.append(token_raw)
            else:
                all_sentences.append(convert_sentence(token_lines))
                token_lines = []
        if len(token_lines) > 0:
            all_sentences.append(convert_sentence(token_lines))
    return all_sentences



def precess_data():
    print('Preparing data to be model-ready ...')

    build_vocab()
    # token2id('cnews.train.txt', 'train')
    # token2id('cnews.test.txt', 'test')
    # token2id('cnews.val.txt', 'val')



















if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/mlp-parser.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    precess_data()