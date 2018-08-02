import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import pickle

from utils import Config, strQ2B

UNK = "<UNK>"
ROOT = "<ROOT>"


def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def _bytes_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(value)))


class Token:
    def __init__(self, token_id, word, pos, dep, head_id):
        self.token_id = token_id
        self.word = word
        self.pos = pos
        self.dep = dep
        self.head_id = head_id
        self.order = None  # order of levelorder traversal
        self.pred_dep_id = None
        self.pred_head_id = None
        self.swapped = []
        self.left_children = []  # for inorder_traversal to detect non-projective
        self.right_children = []  # for inorder_traversal to detect non-projective
        self.comp_head = None
        self.comp_dep = None
        self.comp_rel = None

    def levelorder_traversal(self):

        def recurse(root):
            tokens = [root]
            temp = [root]
            while temp:
                current = temp.pop(0)
                if current.comp_dep:
                    temp.extend([current.comp_dep, current.comp_head])
                    tokens.extend([current.comp_dep, current.comp_head])
            return reversed(tokens)

        return recurse(self)

    def reset_children(self):
        self.left_children = []
        self.right_children = []


class Sentence:
    def __init__(self, tokens):
        self.tokens = tokens
        self.buff = [token for token in self.tokens]
        self.stack = [Token(0, ROOT, ROOT, ROOT, -1)]
        self.history_action = []
        self.terminate = False

    def inorder_traversal(self):
        """get inorder traversal index to convert non-projective to projective"""
        node_dict = {token.token_id: token for token in self.tokens}
        root = None
        for token in self.tokens:
            if token.head_id != 0:
                head_token = node_dict[token.head_id]
                if token.token_id < head_token.token_id:
                    head_token.left_children.append(token)
                else:
                    head_token.right_children.append(token)
            else:
                root = token

        def recurse(root, result):
            """recurse to traverse"""
            for n in root.left_children:
                recurse(n, result)
            result.append(root)
            for n in root.right_children:
                recurse(n, result)

        result = []
        assert root  # wrong data, no root
        recurse(root, result)
        assert len(result) == len(self.tokens)  # wrong data, multiple trees
        for i, token in enumerate(result):
            token.inorder_traversal_idx = i + 1
            token.reset_children()


class ArcStandardParser:
    def __init__(self):
        self.vocab = load_vocab()
        self.pos_dict = load_pos()
        self.dep_dict = load_dep()

    def extract_composition_tree(self, sentence):
        tree_tokens = []
        stack_order = []
        for token in sentence.stack:
            tree_tokens.extend(token.levelorder_traversal())
            stack_order.append(len(tree_tokens) - 1)

        for i, token in enumerate(tree_tokens):
            token.order = i
        comp_head_order, comp_dep_order, comp_rel_id, is_leaf = [], [], [], []
        for token in tree_tokens:
            if token.comp_rel:
                comp_head_order.append(token.comp_head.order)
                comp_dep_order.append(token.comp_dep.order)
                comp_rel_id.append(token.comp_rel)
                is_leaf.append(0)
            else:
                comp_head_order.append(0)
                comp_dep_order.append(0)
                comp_rel_id.append(0)
                is_leaf.append(1)

        return tree_tokens, comp_head_order, comp_dep_order, comp_rel_id, is_leaf, stack_order

    def extract_from_current_state(self, sentence):
        tree_tokens, comp_head_order, comp_dep_order, comp_rel_id, is_leaf, stack_order = self.extract_composition_tree(
            sentence)  # comp trees include stack word
        buff_token = sentence.buff
        history_action_id = sentence.history_action

        tree_word_id = [self.vocab.get(token.word, self.vocab[UNK]) for token in tree_tokens]
        tree_pos_id = [self.pos_dict[token.pos] for token in tree_tokens]
        buff_word_id = [self.vocab.get(token.word, self.vocab[UNK]) for token in buff_token[::-1]]  # reversed
        buff_pos_id = [self.pos_dict[token.pos] for token in buff_token[::-1]]  # reversed

        return tree_word_id, tree_pos_id, buff_word_id, buff_pos_id, history_action_id, comp_head_order, comp_dep_order, comp_rel_id, is_leaf, stack_order

    def get_legal_transitions(self, sentence):
        """check legality of shift, swap, left reduce, right reduce"""
        transitions = [1] if len(sentence.buff) > 0 else [0]
        transitions += [1] if len(sentence.stack) > 2 and sentence.stack[-1] not in sentence.stack[-2].swapped else [0]
        transitions += ([1] if len(sentence.stack) > 2 else [0])
        transitions += ([1] if len(sentence.stack) >= 2 else [0])
        transitions = transitions[:2] + transitions[2:] * (Config.model.dep_num)
        return transitions

    def get_oracle_from_current_state(self, sentence):
        """get oracle according to current state"""
        if len(sentence.stack) < 2:
            return 0  # shift

        stack_token_0 = sentence.stack[-1]
        stack_token_1 = sentence.stack[-2]
        buff_head = [token.head_id for token in sentence.buff]
        # σ[0] is head of σ[1] and all childs of σ[1] are attached to it and σ[1] is not the root
        if stack_token_1.token_id != 0 and stack_token_1.head_id == stack_token_0.token_id \
                and stack_token_1.token_id not in buff_head:
            return 2 + self.dep_dict[stack_token_1.dep] * 2  # left reduce
        # σ[1] is head of σ[0] and all childs of σ[0] are attached to it and σ[0] is not the root
        elif stack_token_0.head_id == stack_token_1.token_id and stack_token_0.token_id not in buff_head:
            return 2 + self.dep_dict[stack_token_0.dep] * 2 + 1  # right reduce
        elif stack_token_1.token_id != 0 and stack_token_1.inorder_traversal_idx > stack_token_0.inorder_traversal_idx:
            return 1  # swap
        else:
            return 0 if len(sentence.buff) != 0 else None

    def update_state_by_transition(self, sentence, transition):
        """updates stack, buffer and dependencies"""
        if transition == 0:  # shift
            sentence.stack.append(sentence.buff[0])
            sentence.buff = sentence.buff[1:] if len(sentence.buff) > 1 else []
        elif transition == 1:  # swap
            token = sentence.stack.pop(-2)
            token.swapped.append(sentence.stack[-1])
            sentence.stack[-1].swapped.append(token)
            sentence.buff.insert(0, token)
        elif transition % 2 == 0:  # left reduce
            sentence.stack.pop(-2)
        elif transition % 2 == 1:  # right reduce
            sentence.stack.pop(-1)
        sentence.history_action.append(transition)

    def update_composition(self, sentence, transition):
        """update composition"""
        if transition % 2 == 0:  # left reduce
            head = sentence.stack[-1]
            dependent = sentence.stack[-2]
        elif transition % 2 == 1:  # right reduce
            head = sentence.stack[-2]
            dependent = sentence.stack[-1]

        dependent.pred_head_id = head.token_id  # store pred head
        dependent.pred_dep_id = (transition - 2) // 2  # store pred dep

        head.comp_head = pickle.loads(pickle.dumps(head, -1))
        head.comp_dep = dependent
        head.comp_rel = transition - 2

    def terminal(self, sentence):
        if len(sentence.stack) == 1 and sentence.stack[0].word == ROOT and sentence.buff == []:
            return True
        else:
            return False


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
    vocab, pos, dep = set(), set(), set()
    sen = []
    total_sentences = []

    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                line = strQ2B(line)
                i, w, _, p, _, _, h, d, _, _ = line.split()
                vocab.add(w)
                pos.add(p)
                dep.add(d)
                sen.append(Token(int(i), w, p, d, int(h)))
            else:
                if 5 <= len(sen) <= 30:
                    total_sentences.append(Sentence(sen))
                sen = []

        if len(sen) > 0:
            total_sentences.append(Sentence(sen))

    with open(vocab_file, 'w', encoding='utf8') as f:
        f.write('\n'.join([UNK, ROOT] + sorted(vocab)))
    with open(pos_file, 'w', encoding='utf8') as f:
        f.write('\n'.join([ROOT] + sorted(pos)))
    with open(dep_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(sorted(dep)))

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
                if 5 <= len(sen) <= 30:
                    total_sentences.append(Sentence(sen))
                sen = []
        if len(sen) > 0:
            total_sentences.append(Sentence(sen))

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
    word_id = [vocab.get(word, vocab[UNK]) for word in words]
    return word_id


def pos2id(pos, dict):
    pos_id = [dict[p] for p in pos]
    return pos_id


def dep2id(dep, dict):
    dep_id = [dict[d] for d in dep]
    return dep_id


def id2word(id, dict):
    id2word = {i: t for i, t in enumerate(dict)}
    return [id2word[i] for i in id]


def id2pos(id, dict):
    id2pos = {i: t for i, t in enumerate(dict)}
    return [id2pos[i] for i in id]


def id2dep(id, dict):
    id2dep = {i: t for i, t in enumerate(dict)}
    return [id2dep[i] for i in id]


def convert_to_train_example(tree_word_id, tree_pos_id, buff_word_id, buff_pos_id, history_action_id, comp_head_order,
                             comp_dep_order, comp_rel_id, is_leaf, stack_order, transition):
    """convert one sample to example"""
    data = {
        'tree_word_id': _int64_feature(tree_word_id),
        'tree_pos_id': _int64_feature(tree_pos_id),
        'buff_word_id': _int64_feature(buff_word_id),
        'buff_pos_id': _int64_feature(buff_pos_id),
        'history_action_id': _int64_feature(history_action_id),
        'comp_head_order': _int64_feature(comp_head_order),
        'comp_dep_order': _int64_feature(comp_dep_order),
        'comp_rel_id': _int64_feature(comp_rel_id),
        'is_leaf': _int64_feature(is_leaf),
        'stack_order': _int64_feature(stack_order),
        'transition': _int64_feature(transition),
        'stack_length': _int64_feature(len(stack_order)),
        'buff_length': _int64_feature(len(buff_word_id)),
        'history_action_length': _int64_feature(len(history_action_id))
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def convert_to_eval_example(word, pos, head, dep_id):
    data = {
        'pos': _bytes_feature(pos),
        'head': _int64_feature(head),
        'dep_id': _int64_feature(dep_id),
        'word': _bytes_feature(word),
        'length': _int64_feature(len(pos))
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def preprocess_train(serialized):
    def parse_tfrecord(serialized):
        features = {
            'tree_word_id': tf.VarLenFeature(tf.int64),
            'tree_pos_id': tf.VarLenFeature(tf.int64),
            'buff_word_id': tf.VarLenFeature(tf.int64),
            'buff_pos_id': tf.VarLenFeature(tf.int64),
            'history_action_id': tf.VarLenFeature(tf.int64),
            'comp_head_order': tf.VarLenFeature(tf.int64),
            'comp_dep_order': tf.VarLenFeature(tf.int64),
            'comp_rel_id': tf.VarLenFeature(tf.int64),
            'is_leaf': tf.VarLenFeature(tf.int64),
            'stack_order': tf.VarLenFeature(tf.int64),
            'transition': tf.FixedLenFeature([], tf.int64),
            'stack_length': tf.FixedLenFeature([], tf.int64),
            'buff_length': tf.FixedLenFeature([], tf.int64),
            'history_action_length': tf.FixedLenFeature([], tf.int64),
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        tree_word_id = tf.sparse_tensor_to_dense(parsed_example['tree_word_id'])
        tree_pos_id = tf.sparse_tensor_to_dense(parsed_example['tree_pos_id'])
        buff_word_id = tf.sparse_tensor_to_dense(parsed_example['buff_word_id'])
        buff_pos_id = tf.sparse_tensor_to_dense(parsed_example['buff_pos_id'])
        history_action_id = tf.sparse_tensor_to_dense(parsed_example['history_action_id'])
        comp_head_order = tf.sparse_tensor_to_dense(parsed_example['comp_head_order'])
        comp_dep_order = tf.sparse_tensor_to_dense(parsed_example['comp_dep_order'])
        comp_rel_id = tf.sparse_tensor_to_dense(parsed_example['comp_rel_id'])
        is_leaf = tf.sparse_tensor_to_dense(parsed_example['is_leaf'])
        stack_order = tf.sparse_tensor_to_dense(parsed_example['stack_order'])
        stack_length = parsed_example['stack_length']
        transition = parsed_example['transition']
        buff_length = parsed_example['buff_length']
        history_action_length = parsed_example['history_action_length']

        return tree_word_id, tree_pos_id, buff_word_id, buff_pos_id, history_action_id, comp_head_order, comp_dep_order, \
               comp_rel_id, is_leaf, stack_order, transition, stack_length, buff_length, history_action_length

    return parse_tfrecord(serialized)


def preprocess_eval(serialized):
    def parse_tfrecord(serialized):
        features = {
            'word': tf.VarLenFeature(tf.string),
            'pos': tf.VarLenFeature(tf.string),
            'head': tf.VarLenFeature(tf.int64),
            'dep_id': tf.VarLenFeature(tf.int64),
            'length': tf.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        word = tf.sparse_tensor_to_dense(parsed_example['word'], default_value='')
        pos = tf.sparse_tensor_to_dense(parsed_example['pos'], default_value='')
        head = tf.sparse_tensor_to_dense(parsed_example['head'])
        dep_id = tf.sparse_tensor_to_dense(parsed_example['dep_id'])
        length = parsed_example['length']
        return word, pos, head, dep_id, length

    return parse_tfrecord(serialized)


def get_train_batch(data, buffer_size=1, batch_size=64, scope="train"):
    class IteratorInitializerHook(tf.train.SessionRunHook):

        def __init__(self):
            self.iterator_initializer_func = None

        def after_create_session(self, session, coord):
            self.iterator_initializer_func(session)

    iterator_initializer_hook = IteratorInitializerHook()

    def inputs():
        input_placeholder = tf.placeholder(tf.string)
        dataset = tf.data.TFRecordDataset(input_placeholder)
        dataset = dataset.map(preprocess_train)

        if scope == "train":
            dataset = dataset.repeat(None)  # Infinite iterations
        else:
            dataset = dataset.repeat(1)  # 1 Epoch
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.padded_batch(batch_size, ([-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1],
                                                    [], [], [], []))
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next('next_batch')
        tree_word_id = next_batch[0]
        tree_pos_id = next_batch[1]
        buff_word_id = next_batch[2]
        buff_pos_id = next_batch[3]
        history_action_id = next_batch[4]
        comp_head_order = next_batch[5]
        comp_dep_order = next_batch[6]
        comp_rel_id = next_batch[7]
        is_leaf = next_batch[8]
        stack_order = next_batch[9]
        transition = next_batch[10]
        stack_length = next_batch[11]
        buff_length = next_batch[12]
        history_action_length = next_batch[13]

        iterator_initializer_hook.iterator_initializer_func = \
            lambda sess: sess.run(
                iterator.initializer,
                feed_dict={input_placeholder: np.random.permutation(data)})

        return {'tree_word_id': tree_word_id, 'tree_pos_id': tree_pos_id, 'buff_word_id': buff_word_id,
                'buff_pos_id': buff_pos_id, 'history_action_id': history_action_id,
                'comp_head_order': comp_head_order, 'comp_dep_order': comp_dep_order, 'comp_rel_id': comp_rel_id,
                'is_leaf': is_leaf, 'stack_order': stack_order, 'stack_length': stack_length,
                'buff_length': buff_length, 'history_action_length': history_action_length}, {'transition': transition}

    return inputs, iterator_initializer_hook


def get_eval_batch(data, buffer_size=1, batch_size=64):
    class IteratorInitializerHook(tf.train.SessionRunHook):
        def __init__(self):
            self.iterator_initializer_func = None

        def after_create_session(self, session, coord):
            self.iterator_initializer_func(session)

    iterator_initializer_hook = IteratorInitializerHook()

    def inputs():
        input_placeholder = tf.placeholder(tf.string)
        dataset = tf.data.TFRecordDataset(input_placeholder)
        dataset = dataset.map(preprocess_eval)

        dataset = dataset.repeat(1)  # 1 Epoch
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.padded_batch(batch_size, ([-1], [-1], [-1], [-1], []))
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next('next_batch')
        word = next_batch[0]
        pos = next_batch[1]
        head = next_batch[2]
        dep_id = next_batch[3]
        length = next_batch[4]

        iterator_initializer_hook.iterator_initializer_func = \
            lambda sess: sess.run(
                iterator.initializer,
                feed_dict={input_placeholder: data})

        return {'word': word, 'pos': pos, 'length': length}, {'head': head, 'dep_id': dep_id}

    return inputs, iterator_initializer_hook


def create_tfrecord():
    train_file = os.path.join(Config.data.dataset_path, Config.data.train_data)
    train_data = build_and_read_train(train_file)
    test_file = os.path.join(Config.data.dataset_path, Config.data.test_data)
    test_data = read_test(test_file)
    build_wordvec_pkl()
    pos_dict = load_pos()
    dep_dict = load_dep()
    parser = ArcStandardParser()

    if len(pos_dict) != Config.model.pos_num:
        raise ValueError('pos_num should be %d' % len(pos_dict))
    if len(dep_dict) != Config.model.dep_num:
        raise ValueError('dep_num should be %d' % len(dep_dict))

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
                    sys.stdout.write('\r>> converting %d/%d' % (i + 1, len(data)))
                    sys.stdout.flush()
                    sen = data[i]
                    try:
                        sen.inorder_traversal()
                    except AssertionError:
                        print('\nskip wrong data %d' % (i + 1))
                        i += 1
                        continue
                    if data == train_data:
                        while True:
                            tree_word_id, tree_pos_id, buff_word_id, buff_pos_id, history_action_id, comp_head_order, \
                            comp_dep_order, comp_rel_id, is_leaf, stack_order \
                                = parser.extract_from_current_state(sen)
                            legal_transitions = parser.get_legal_transitions(sen)
                            transition = parser.get_oracle_from_current_state(sen)
                            if transition is None:
                                print('\nerror')
                                continue
                            assert legal_transitions[transition] == 1, 'oracle is illegal'

                            example = convert_to_train_example(tree_word_id, tree_pos_id, buff_word_id, buff_pos_id,
                                                               history_action_id, comp_head_order, comp_dep_order,
                                                               comp_rel_id, is_leaf, stack_order, transition)
                            serialized = example.SerializeToString()
                            tfrecord_writer.write(serialized)
                            j += 1

                            if transition not in [0, 1]:
                                parser.update_composition(sen, transition)  # update composition
                            parser.update_state_by_transition(sen, transition)  # update stack and buff

                            if parser.terminal(sen):
                                break
                        i += 1
                        if j >= 5000:  # totally shuffled
                            break
                    else:

                        word = [t.word.encode() for t in sen.tokens]
                        pos = [t.pos.encode() for t in sen.tokens]
                        head = [t.head_id for t in sen.tokens]
                        dep_id = dep2id([t.dep for t in sen.tokens], dep_dict)
                        example = convert_to_eval_example(word, pos, head, dep_id)
                        serialized = example.SerializeToString()
                        tfrecord_writer.write(serialized)
                        i += 1
                fidx += 1
            print('\n%s complete' % tf_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/stack-lstm+swap.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    Config.data.dataset_path = os.path.expanduser(Config.data.dataset_path)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    create_tfrecord()
