import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import pickle

from utils import Config, strQ2B

NULL = "<NULL>"
UNK = "<UNK>"
ROOT = "<ROOT>"


# pos_prefix = "<p>:"
# dep_prefix = "<d>:"
# punc_pos = ["''", "``", ":", ".", ","]



def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


class Token:
    def __init__(self, token_id, word, pos, dep, head_id):
        self.token_id = token_id
        self.word = word.lower()
        self.pos = pos
        self.dep = dep
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



class ArcStandardParser:
    def __init__(self,tokens):
        self.Root = ROOT_TOKEN
        self.tokens = tokens
        self.buff = [token for token in self.tokens]
        self.stack = [self.Root]
        self.dependencies = []
        self.predicted_dependencies = []

    def extract_from_stack_and_buffer(self, num_word=3):
        """extract the last 3 tokens in the stack and the first 3 tokens in the buff, concat them as direct_tokens"""
        tokens = []
        # pad NULL at the beginning until 3 if len(stack) < 3
        tokens.extend([NULL_TOKEN for _ in range(num_word - len(self.stack))])
        # extract the last 3 tokens in stack
        tokens.extend(self.stack[-num_word:])
        # add the first 3 tokens in buff
        tokens.extend(self.buff[:num_word])
        # pad NULL at the end until 3 if len(buff) < 3
        tokens.extend([NULL_TOKEN for _ in range(num_word - len(self.buff))])
        # return a list of 6 tokens
        return tokens

    def extract_for_current_state(self, word_vocab, pos_vocab, dep_vocab):
        """cal direct_tokens and children_tokens to combine current state"""
        direct_tokens = self.extract_from_stack_and_buffer(Config.data.num_stack_word)  # 6 tokens
        children_tokens = self.extract_children_from_stack(Config.data.children_stack_range)  # 12 tokens

        word_features = []
        pos_features = []
        dep_features = []

        # Word features -> 18
        word_features.extend([token.word for token in direct_tokens])
        word_features.extend([token.word for token in children_tokens])

        # pos features -> 18
        pos_features.extend([token.pos for token in direct_tokens])
        pos_features.extend([token.pos for token in children_tokens])

        # dep features -> 12 (only children)
        dep_features.extend([token.dep for token in children_tokens])

        word_input_ids = [word_vocab.get(word, word_vocab[UNK_TOKEN.word]) for word in word_features]
        pos_input_ids = [pos_vocab.get(pos, pos_vocab[UNK_TOKEN.pos]) for pos in pos_features]
        dep_input_ids = [dep_vocab.get(dep, dep_vocab[UNK_TOKEN.dep]) for dep in dep_features]

        return [word_input_ids, pos_input_ids, dep_input_ids]  # 48 features

    def extract_children_from_stack(self, num_word=2):
        """extract children from the last 2 token in stack"""
        children_tokens = []
        for i in range(num_word):
            if len(self.stack) > i:
                # the first token in the token.left_children
                lc0 = self.get_child_by_index_and_depth(self.stack[-i - 1], 0, "left", 1)
                # the first token in the token.right_children
                rc0 = self.get_child_by_index_and_depth(self.stack[-i - 1], 0, "right", 1)
                # the second token in the token.left_children
                lc1 = self.get_child_by_index_and_depth(self.stack[-i - 1], 1, "left",
                                                            1) if lc0 != NULL_TOKEN else NULL_TOKEN
                # the second token in the token.right_children
                rc1 = self.get_child_by_index_and_depth(self.stack[-i - 1], 1, "right",
                                                            1) if rc0 != NULL_TOKEN else NULL_TOKEN
                # the first token in the left_children of the first token in the token.left_children
                llc0 = self.get_child_by_index_and_depth(self.stack[-i - 1], 0, "left",
                                                             2) if lc0 != NULL_TOKEN else NULL_TOKEN
                # the first token in the right_children of the first token in the token.right_children
                rrc0 = self.get_child_by_index_and_depth(self.stack[-i - 1], 0, "right",
                                                             2) if rc0 != NULL_TOKEN else NULL_TOKEN

                children_tokens.extend([lc0, rc0, lc1, rc1, llc0, rrc0])
            else:
                [children_tokens.append(NULL_TOKEN) for _ in range(6)]
        # return 12 tokens
        return children_tokens

    def get_child_by_index_and_depth(self, token, index, direction, depth):
        """get child token, return NULL if no child"""
        if depth == 0:
            return token

        if direction == "left":
            if len(token.left_children) > index:
                return self.get_child_by_index_and_depth(
                    self.tokens[token.left_children[index] - 1], index, direction, depth - 1)
            return NULL_TOKEN
        else:
            if len(token.right_children) > index:
                return self.get_child_by_index_and_depth(
                    self.tokens[token.right_children[::-1][index] - 1], index, direction, depth - 1)
            return NULL_TOKEN

    def get_legal_labels(self):
        """check legality of shift, left arc, right arc"""
        labels = [1] if len(self.buff) > 0 else [0]
        labels += ([1] if len(self.stack) > 2 else [0])
        labels += ([1] if len(self.stack) >= 2 else [0])
        labels = [labels[0]] + labels[1:] * Config.data.num_dep
        return labels

    def get_transition_from_current_state(self, dep_dict):
        """get transition according to stack0 and stack1"""
        if len(self.stack) < 2:
            return 0  # shift

        stack_token_0 = self.stack[-1]
        stack_token_1 = self.stack[-2]
        if stack_token_1.token_id >= 1 and stack_token_1.head_id == stack_token_0.token_id:
            return dep_dict[stack_token_1.dep] * 2 + 1  # left arc
        elif stack_token_1.token_id >= 0 and stack_token_0.head_id == stack_token_1.token_id \
                and stack_token_0.token_id not in map(lambda x: x.head_id, self.buff):
            return dep_dict[stack_token_0.dep] * 2 + 2  # right arc
        else:
            return 0 if len(self.buff) != 0 else None

    def update_state_by_transition(self, transition, gt=True):
        """updates stack, buffer and dependencies"""
        if transition is not None:
            if transition == 0:  # shift
                self.stack.append(self.buff[0])
                self.buff = self.buff[1:] if len(self.buff) > 1 else []
            elif transition % 2 == 1:  # left arc
                # save in self.dependencies
                self.dependencies.append(
                    (self.stack[-1], self.stack[-2])) if gt else self.predicted_dependencies.append(
                    (self.stack[-1], self.stack[-2], transition))
                # del the children token
                self.stack = self.stack[:-2] + self.stack[-1:]
            elif transition % 2 == 0:  # right arc
                self.dependencies.append(
                    (self.stack[-2], self.stack[-1])) if gt else self.predicted_dependencies.append(
                    (self.stack[-2], self.stack[-1], transition))
                self.stack = self.stack[:-1]

    def clear_prediction_dependencies(self):
        self.predicted_dependencies = []

    def clear_children_info(self):
        for token in self.tokens:
            token.left_children = []
            token.right_children = []

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
        """update left/right children"""
        if curr_transition % 2 == 1:
            head = self.stack[-1]
            dependent = self.stack[-2]
        elif curr_transition % 2 == 0:
            head = self.stack[-2]
            dependent = self.stack[-1]

        if head.token_id > dependent.token_id:
            head.left_children.append(dependent.token_id)
            head.left_children.sort()
        else:
            head.right_children.append(dependent.token_id)
            head.right_children.sort()



def get_tfrecord(name):
    tfrecords = []
    files = os.listdir(os.path.join(Config.data.processed_path, 'tfrecord/'))
    for file in files:
        if name in file and file.endswith('.tfrecord'):
            tfrecords.append(os.path.join(Config.data.processed_path, 'tfrecord/', file))
    if len(tfrecords) == 0:
        raise RuntimeError('TFrecord not found.')

    return tfrecords


def build_and_read_train(file):
    vocab_file = os.path.join(Config.data.processed_path, Config.data.vocab_file)
    pos_file = os.path.join(Config.data.processed_path, Config.data.pos_file)
    dep_file = os.path.join(Config.data.processed_path, Config.data.dep_file)
    vocab, pos_tag, dep_tag = set(), set(), set()
    sen = []
    total_sentences = []

    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                line = strQ2B(line)
                i, w, _, p, _, _, h, d, _, _ = line.split()
                vocab.add(w)
                pos_tag.add(p)
                dep_tag.add(d)
                sen.append(Token(int(i), w, p, d, int(h)))
            else:
                total_sentences.append(sen)
                sen = []
        if len(sen) > 0:
            total_sentences.append(sen)

    with open(vocab_file, 'w', encoding='utf8') as f:
        f.write('\n'.join([NULL, UNK, ROOT] + sorted(vocab)))
    with open(pos_file, 'w', encoding='utf8') as f:
        f.write('\n'.join([NULL, ROOT] + sorted(pos_tag)))
    with open(dep_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(sorted(dep_tag)))

    return total_sentences


def read_test(file):
    sen = []
    total_sentences = []
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                line = strQ2B(line)
                i, w, _, p, _, _, h, d, _, _ = line.split()
                sen.append(Token(int(i), w, p, d, int(h)))
            else:
                total_sentences.append(sen)
                sen = []
        if len(sen) > 0:
            total_sentences.append(sen)

    return total_sentences


def load_pos():
    tag_file = os.path.join(Config.data.processed_path, Config.data.pos_file)
    with open(tag_file, encoding='utf8') as f:
        tag = f.read().splitlines()
    return {t: i for i, t in enumerate(tag)}


def load_dep():
    tag_file = os.path.join(Config.data.processed_path, Config.data.dep_file)
    with open(tag_file, encoding='utf8') as f:
        tag = f.read().splitlines()
    return {t: i for i, t in enumerate(tag)}


def load_vocab():
    vocab_file = os.path.join(Config.data.processed_path, Config.data.vocab_file)
    with open(vocab_file, encoding='utf8') as f:
        words = f.read().splitlines()
    return {word: i for i, word in enumerate(words)}


def build_wordvec_pkl():
    file = os.path.join(Config.data.processed_path, Config.data.wordvec_file)
    vocab = load_vocab()
    vocab_size = len(vocab)
    wordvec = np.zeros([vocab_size, Config.model.word_embedding_size])

    with open(file, encoding='utf8') as f:
        wordvec_dict = {}
        print('It may take some time to load wordvec file ...')
        for i, line in enumerate(f):
            if len(line.split()) < Config.model.word_embedding_size + 1:
                continue
            word = line.strip().split(' ')[0]
            vec = line.strip().split(' ')[1:]
            if word in vocab:
                wordvec_dict[word] = vec

    for index, word in enumerate(vocab):
        if word in wordvec_dict:
            wordvec[index] = wordvec_dict[word]
        else:
            wordvec[index] = np.random.rand(Config.model.word_embedding_size)

    with open(os.path.join(Config.data.processed_path, Config.data.wordvec_pkl), 'wb') as f:
        pickle.dump(wordvec, f)


def load_pretrained_vec():
    file = os.path.join(Config.data.processed_path, Config.data.wordvec_pkl)
    with open(file, 'rb') as f:
        wordvec = pickle.load(f)
    return wordvec


def word2id(words, vocab):
    word_id = [vocab.get(word, vocab['<UNK>']) for word in words]
    return word_id


def pos2id(pos, dict):
    pos_id = [dict[p] for p in pos]
    return pos_id


def dep2id(dep, dict):
    dep_id = [dict[d] for d in dep]
    return dep_id


def id2dep(id, dict):
    id2dep = {i: t for i, t in enumerate(dict)}
    return [id2dep[i] for i in id]


def create_tfrecord():
    train_file = os.path.join(Config.data.dataset_path, Config.data.train_data)
    train_data = build_and_read_train(train_file)
    test_file = os.path.join(Config.data.dataset_path, Config.data.test_data)
    test_data = read_test(test_file)
    build_wordvec_pkl()
    vocab = load_vocab()
    pos_dict = load_pos()
    dep_dict = load_dep()

    if len(pos_dict) != Config.model.pos_num:
        raise ValueError('length of pos dict must be as same as pos_num')
    if len(dep_dict) != Config.model.dep_num:
        raise ValueError('length of dep dict must be as same as dep_num')

    if not os.path.exists(os.path.join(Config.data.processed_path, 'tfrecord')):
        os.makedirs(os.path.join(Config.data.processed_path, 'tfrecord'))

    print('writing to tfrecord ...')
    for data in [train_data, test_data]:
        i = 0
        fidx = 0
        while i < len(data):
            if data == train_data:
                tf_file = 'train_%d.tfrecord' % fidx
            else:
                tf_file = 'test.tfrecord'
            tf_file = os.path.join(Config.data.processed_path, 'tfrecord', tf_file)
            with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
                j = 0
                while i < len(data):
                    sys.stdout.write('\r>> converting %d/%d' % (i + 1, len(sen)))
                    sys.stdout.flush()
                    parser = ArcStandardParser(data[i])
                    num_word = len(parser.tokens)
                    for _ in range(num_word * 2):
                        legal_labels = parser.get_legal_labels()
                        curr_transition = parser.get_transition_from_current_state(dep_dict)
                        # non-projective
                        if curr_transition is None:
                            break
                        assert legal_labels[curr_transition] == 1
                        # update left/right children
                        if curr_transition != 0:
                            sentence.update_child_dependencies(curr_transition)
                        # update stack
                        sentence.update_state_by_transition(curr_transition)

                    serialized = example.SerializeToString()
                    tfrecord_writer.write(serialized)
                    i += 1
                    j += 1
                    if j >= 5000 and data == train_data:  # totally shuffled
                        break
                fidx += 1
            print('\n%s complete' % tf_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/structured-prediction.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    Config.data.dataset_path = os.path.expanduser(Config.data.dataset_path)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    create_tfrecord()
