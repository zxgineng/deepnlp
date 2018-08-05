import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import pickle

from utils import Config, strQ2B

UNK = "<UNK>"
ROOT = "<ROOT>"
NULL = "<NULL>"


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
        self.pred_dep_id = []  # multi-head in graph
        self.pred_head_id = []
        self.children = []

    def levelorder_traversal(self):

        def recurse(root):
            tokens = [root]
            temp = [root]
            while temp:
                current = temp.pop(0)
                for c in current.children:
                    temp.append(c)
                    tokens.append(c)
            return reversed(tokens)

        return recurse(self)


class Sentence:
    def __init__(self, tokens):
        self.tokens = tokens
        self.buff_top_id = 0
        self.deque = []
        self.stack = [Token(0, ROOT, ROOT, [], [])]
        self.history_action = [0]  # start with null
        self.terminate = False


class ArcEagerParser:
    def __init__(self):
        self.vocab = load_vocab()
        self.pos_dict = load_pos()
        self.dep_dict = load_dep()

    def extract_stack_tree(self, sentence):
        tree_tokens = []
        stack_order = []  # store the index of stack word in flatten tree
        stack = [Token(-1, NULL, NULL, [], [])] + sentence.stack  # add null

        for token in stack:
            tree_tokens.extend(token.levelorder_traversal())
            stack_order.append(len(tree_tokens))  # the 0 place is preserved for 0.0(no children state)

        for i, token in enumerate(reversed(tree_tokens)):
            token.order = len(tree_tokens) - i
        children_order = []
        for token in tree_tokens:
            children_order.append([t.order for t in token.children])

        return tree_tokens, children_order, stack_order

    def extract_from_current_state(self, sentence):
        tree_tokens, children_order, stack_order = self.extract_stack_tree(sentence)
        history_action_id = sentence.history_action
        buff_top_id = sentence.buff_top_id
        deque = [Token(-1, NULL, NULL, [], [])] + sentence.deque  # add null

        tree_word_id = [self.vocab.get(token.word, self.vocab[UNK]) for token in tree_tokens]
        tree_pos_id = [self.pos_dict[token.pos] for token in tree_tokens]
        token_word_id = [self.vocab.get(token.word, self.vocab[UNK]) for token in sentence.tokens]
        token_pos_id = [self.pos_dict[token.pos] for token in sentence.tokens]
        deque_word_id = [self.vocab.get(token.word, self.vocab[UNK]) for token in deque]
        deque_pos_id = [self.pos_dict[token.pos] for token in deque]

        return tree_word_id, tree_pos_id, token_word_id, token_pos_id, buff_top_id, history_action_id, \
               deque_word_id, deque_pos_id, children_order, stack_order

    def get_legal_transitions(self, sentence):
        """check legality of no-shift, no-reduce, no-pass, left-reduce, left-pass, right-shift, right-pass"""
        wi = sentence.stack[-1]
        wj = sentence.tokens[sentence.buff_top_id]
        left_legal = wi.token_id != 0 and wj.token_id not in [t.token_id for t in wi.levelorder_traversal()]
        right_legal = wi.token_id not in [t.token_id for t in wj.levelorder_traversal()]
        reduce_legal = wi.pred_head_id

        transitions = [0]  # NULL
        transitions += [1]  # no-shift
        transitions += [1] if reduce_legal else [0]  # no-reduce
        transitions += [1] if wi.token_id != 0 else [0]  # no-pass
        transitions += [1] if left_legal else [0]  # left-reduce
        transitions += [1] if left_legal else [0]  # left-pass
        transitions += [1] if right_legal else [0]  # right-shift
        transitions += [1] if right_legal and wi.token_id != 0 else [0]  # right-pass
        transitions = transitions[:4] + transitions[4:] * (Config.model.dep_num)

        return transitions

    def get_oracle_from_current_state(self, sentence):
        """get oracle according to current state"""
        buff = sentence.tokens[sentence.buff_top_id:]
        wi = sentence.stack[-1]
        wj = buff[0]
        reduce_legal = not (wi.token_id in sum([t.head_id for t in buff[1:]], []) or
                            (set(wi.head_id) & {t.token_id for t in buff[1:]})) \
            if len(buff) > 1 else True
        shift_legal = not (wj.token_id in sum([t.head_id for t in sentence.stack[1:-1]], []) or
                           (set(wj.head_id) & {t.token_id for t in sentence.stack[1:-1]})) \
            if len(sentence.stack) > 2 else True

        if (wj.token_id not in wi.head_id) and (wi.token_id not in wj.head_id):  # no-*
            if reduce_legal and wi.pred_head_id:
                return 2  # no-reduce
            elif shift_legal:
                return 1  # no-shift
            else:
                return 3  # no-pass
        elif wj.token_id in wi.head_id:  # left-*
            if reduce_legal:
                return 4 + (self.dep_dict[wi.dep[wi.head_id.index(wj.token_id)]]) * 4  # left-reduce
            else:
                return 4 + (self.dep_dict[wi.dep[wi.head_id.index(wj.token_id)]]) * 4 + 1  # left-pass
        else:  # right-*
            if shift_legal:
                return 4 + (self.dep_dict[wj.dep[wj.head_id.index(wi.token_id)]]) * 4 + 2  # right-shift
            else:
                return 4 + (self.dep_dict[wj.dep[wj.head_id.index(wi.token_id)]]) * 4 + 3  # right-pass

    def update_state_by_transition(self, sentence, transition):
        """updates stack and buffer"""
        if transition == 1:  # no-shift
            sentence.stack.extend(sentence.deque + [sentence.tokens[sentence.buff_top_id]])
            sentence.deque = []
            sentence.buff_top_id += 1
        elif transition == 2:  # no-reduce
            sentence.stack.pop(-1)
        elif transition == 3:  # no-pass
            temp = sentence.stack.pop(-1)
            sentence.deque.insert(0, temp)
        elif (transition - 4) % 4 == 0:  # left-reduce
            sentence.stack.pop(-1)
        elif (transition - 4) % 4 == 1:  # left-pass
            temp = sentence.stack.pop(-1)
            sentence.deque.insert(0, temp)
        elif (transition - 4) % 4 == 2:  # right-shift
            sentence.stack.extend(sentence.deque + [sentence.tokens[sentence.buff_top_id]])
            sentence.deque = []
            sentence.buff_top_id += 1
        else:  # right-pass
            temp = sentence.stack.pop(-1)
            sentence.deque.insert(0, temp)

        sentence.history_action.append(transition)

    def update_tree(self, sentence, transition):
        """update tree"""
        if (transition - 4) % 4 == 0:  # left-reduce
            head = sentence.tokens[sentence.buff_top_id]
            dependent = sentence.stack[-1]
        elif (transition - 4) % 4 == 1:  # left-pass
            head = sentence.tokens[sentence.buff_top_id]
            dependent = sentence.stack[-1]
        elif (transition - 4) % 4 == 2:  # right-shift
            head = sentence.stack[-1]
            dependent = sentence.tokens[sentence.buff_top_id]
        else:  # right-pass
            head = sentence.stack[-1]
            dependent = sentence.tokens[sentence.buff_top_id]

        dependent.pred_head_id.append(head.token_id)  # store pred head
        dependent.pred_dep_id.append((transition - 4) // 4)  # store pred dep
        head.children.append(dependent)

    def terminal(self, sentence):
        if sentence.buff_top_id == len(sentence.tokens):
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


def build_and_read_train(files):
    vocab_file = os.path.join(Config.data.processed_path, Config.data.vocab_file)
    pos_file = os.path.join(Config.data.processed_path, Config.data.pos_file)
    dep_file = os.path.join(Config.data.processed_path, Config.data.dep_file)
    vocab, pos, dep = set(), set(), set()
    total_sentences = []
    for file in files:
        sen = []
        with open(file, encoding='utf8') as f:
            temp_token = Token(-1, NULL, NULL, None, None)
            for line in f:
                line = line.strip()
                if line:
                    line = strQ2B(line)
                    i, w, _, p, _, _, h, d, _, _ = line.split()
                    vocab.add(w)
                    pos.add(p)
                    dep.add(d)
                    new_token = Token(int(i), w, p, [d], [int(h)])
                    if new_token.token_id == temp_token.token_id:
                        sen[-1].head_id.append(int(h))
                        sen[-1].dep.append(d)
                    else:
                        sen.append(new_token)
                    temp_token = new_token
                else:
                    total_sentences.append(Sentence(sen))
                    sen = []
                    temp_token = Token(-1, NULL, NULL, None, None)

            if len(sen) > 0:
                total_sentences.append(Sentence(sen))

    with open(vocab_file, 'w', encoding='utf8') as f:
        f.write('\n'.join([NULL, UNK, ROOT] + sorted(vocab)))
    with open(pos_file, 'w', encoding='utf8') as f:
        f.write('\n'.join([NULL, ROOT] + sorted(pos)))
    with open(dep_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(sorted(dep)))

    return total_sentences


def read_test(files):
    total_sentences = []
    for file in files:
        sen = []
        with open(file, encoding='utf8') as f:
            temp_token = Token(-1, NULL, NULL, None, None)
            for line in f:
                line = line.strip()
                if line:
                    line = strQ2B(line)
                    i, w, _, p, _, _, h, d, _, _ = line.split()
                    new_token = Token(int(i), w, p, [d], [int(h)])
                    if new_token.token_id == temp_token.token_id:
                        sen[-1].head_id.append(int(h))
                        sen[-1].dep.append(d)
                    else:
                        sen.append(new_token)
                    temp_token = new_token
                else:
                    total_sentences.append(Sentence(sen))
                    sen = []
                    temp_token = Token(-1, NULL, NULL, None, None)

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


def id2dep(id, dict):
    id2dep = {i: t for i, t in enumerate(dict)}
    return [id2dep[i] for i in id]


def convert_to_train_example(tree_word_id, tree_pos_id, token_word_id, token_pos_id, buff_top_id, history_action_id,
                             deque_word_id, deque_pos_id, children_order, stack_order, transition):
    """convert one sample to example"""
    children_order = tf.keras.preprocessing.sequence.pad_sequences(children_order, dtype='int64', padding='post')

    data = {
        'tree_word_id': _int64_feature(tree_word_id),
        'tree_pos_id': _int64_feature(tree_pos_id),
        'token_word_id': _int64_feature(token_word_id),
        'token_pos_id': _int64_feature(token_pos_id),
        'buff_top_id': _int64_feature(buff_top_id),
        'history_action_id': _int64_feature(history_action_id),
        'deque_word_id': _int64_feature(deque_word_id),
        'deque_pos_id': _int64_feature(deque_pos_id),
        'deque_length': _int64_feature(len(deque_word_id)),
        'children_order': _bytes_feature(children_order.tostring()),
        'children_num': _int64_feature(children_order.shape[1]),
        'stack_order': _int64_feature(stack_order),
        'transition': _int64_feature(transition),
        'stack_length': _int64_feature(len(stack_order)),
        'token_length': _int64_feature(len(token_word_id)),
        'history_action_length': _int64_feature(len(history_action_id))
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def convert_to_eval_example(word, pos, head, dep_id):
    head = tf.keras.preprocessing.sequence.pad_sequences(head, dtype='int64', padding='post',
                                                         value=-1)
    dep_id = tf.keras.preprocessing.sequence.pad_sequences(dep_id, dtype='int64', padding='post',
                                                           value=-1)
    data = {
        'pos': _bytes_feature(pos),
        'head': _bytes_feature(head.tostring()),
        'head_num': _int64_feature(head.shape[1]),
        'dep_id': _bytes_feature(dep_id.tostring()),
        'dep_num': _int64_feature(dep_id.shape[1]),
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
            'token_word_id': tf.VarLenFeature(tf.int64),
            'token_pos_id': tf.VarLenFeature(tf.int64),
            'history_action_id': tf.VarLenFeature(tf.int64),
            'buff_top_id': tf.FixedLenFeature([], tf.int64),
            'deque_word_id': tf.VarLenFeature(tf.int64),
            'deque_pos_id': tf.VarLenFeature(tf.int64),
            'deque_length': tf.FixedLenFeature([], tf.int64),
            'children_order': tf.FixedLenFeature([], tf.string),
            'children_num': tf.FixedLenFeature([], tf.int64),
            'stack_order': tf.VarLenFeature(tf.int64),
            'transition': tf.FixedLenFeature([], tf.int64),
            'stack_length': tf.FixedLenFeature([], tf.int64),
            'token_length': tf.FixedLenFeature([], tf.int64),
            'history_action_length': tf.FixedLenFeature([], tf.int64),
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        tree_word_id = tf.sparse_tensor_to_dense(parsed_example['tree_word_id'])
        tree_pos_id = tf.sparse_tensor_to_dense(parsed_example['tree_pos_id'])
        token_word_id = tf.sparse_tensor_to_dense(parsed_example['token_word_id'])
        token_pos_id = tf.sparse_tensor_to_dense(parsed_example['token_pos_id'])
        history_action_id = tf.sparse_tensor_to_dense(parsed_example['history_action_id'])
        buff_top_id = parsed_example['buff_top_id']
        deque_word_id = tf.sparse_tensor_to_dense(parsed_example['deque_word_id'])
        deque_pos_id = tf.sparse_tensor_to_dense(parsed_example['deque_pos_id'])
        deque_length = parsed_example['deque_length']
        children_order = tf.decode_raw(parsed_example['children_order'], tf.int64)
        children_num = parsed_example['children_num']
        stack_order = tf.sparse_tensor_to_dense(parsed_example['stack_order'])
        stack_length = parsed_example['stack_length']
        transition = parsed_example['transition']
        token_length = parsed_example['token_length']
        history_action_length = parsed_example['history_action_length']

        return tree_word_id, tree_pos_id, token_word_id, token_pos_id, history_action_id, buff_top_id, deque_word_id, \
               deque_pos_id, deque_length, children_order, children_num, stack_order, transition, stack_length, \
               token_length, history_action_length

    tree_word_id, tree_pos_id, token_word_id, token_pos_id, history_action_id, buff_top_id, deque_word_id, \
    deque_pos_id, deque_length, children_order, children_num, stack_order, transition, stack_length, token_length, \
    history_action_length = parse_tfrecord(serialized)
    children_order = tf.reshape(children_order, tf.stack([tf.shape(tree_word_id, out_type=tf.int64)[0], children_num]))

    return tree_word_id, tree_pos_id, token_word_id, token_pos_id, history_action_id, buff_top_id, deque_word_id, \
           deque_pos_id, deque_length, children_order, stack_order, transition, stack_length, token_length, \
           history_action_length


def preprocess_eval(serialized):
    def parse_tfrecord(serialized):
        features = {
            'word': tf.VarLenFeature(tf.string),
            'pos': tf.VarLenFeature(tf.string),
            'head': tf.FixedLenFeature([], tf.string),
            'head_num': tf.FixedLenFeature([], tf.int64),
            'dep_id': tf.FixedLenFeature([], tf.string),
            'dep_num': tf.FixedLenFeature([], tf.int64),
            'length': tf.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        word = tf.sparse_tensor_to_dense(parsed_example['word'], default_value='')
        pos = tf.sparse_tensor_to_dense(parsed_example['pos'], default_value='')
        head = tf.decode_raw(parsed_example['head'], tf.int64)
        head_num = parsed_example['head_num']
        dep_id = tf.decode_raw(parsed_example['dep_id'], tf.int64)
        dep_num = parsed_example['dep_num']
        length = parsed_example['length']
        return word, pos, head, head_num, dep_id, dep_num, length

    word, pos, head, head_num, dep_id, dep_num, length = parse_tfrecord(serialized)
    head = tf.reshape(head, tf.stack([tf.shape(word, out_type=tf.int64)[0], head_num]))
    dep_id = tf.reshape(dep_id, tf.stack([tf.shape(word, out_type=tf.int64)[0], dep_num]))
    return word, pos, head, dep_id, length


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
        dataset = dataset.padded_batch(batch_size, ([-1], [-1], [-1], [-1], [-1], [], [-1], [-1], [], [-1, -1], [-1],
                                                    [], [], [], []))
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next('next_batch')
        tree_word_id = next_batch[0]
        tree_pos_id = next_batch[1]
        token_word_id = next_batch[2]
        token_pos_id = next_batch[3]
        history_action_id = next_batch[4]
        buff_top_id = next_batch[5]
        deque_word_id = next_batch[6]
        deque_pos_id = next_batch[7]
        deque_length = next_batch[8]
        children_order = next_batch[9]
        stack_order = next_batch[10]
        transition = next_batch[11]
        stack_length = next_batch[12]
        token_length = next_batch[13]
        history_action_length = next_batch[14]

        iterator_initializer_hook.iterator_initializer_func = \
            lambda sess: sess.run(
                iterator.initializer,
                feed_dict={input_placeholder: np.random.permutation(data)})

        return {'tree_word_id': tree_word_id, 'tree_pos_id': tree_pos_id, 'token_word_id': token_word_id,
                'token_pos_id': token_pos_id, 'history_action_id': history_action_id,
                'buff_top_id': buff_top_id, 'deque_word_id': deque_word_id, 'deque_pos_id': deque_pos_id,
                'deque_length': deque_length, 'children_order': children_order, 'stack_order': stack_order,
                'stack_length': stack_length, 'token_length': token_length,
                'history_action_length': history_action_length}, {'transition': transition}

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
        dataset = dataset.padded_batch(batch_size, ([-1], [-1], [-1, -1], [-1, -1], []),
                                       ('', '', tf.cast(-1,tf.int64), tf.cast(-1,tf.int64), tf.cast(0,tf.int64)))
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
    train_files = [os.path.join(Config.data.dataset_path, file) for file in Config.data.train_data]
    train_data = build_and_read_train(train_files)
    test_files = [os.path.join(Config.data.dataset_path, file) for file in Config.data.test_data]
    test_data = read_test(test_files)
    # build_wordvec_pkl()
    pos_dict = load_pos()
    dep_dict = load_dep()
    parser = ArcEagerParser()

    assert len(pos_dict) == Config.model.pos_num, 'pos_num should be %d' % len(pos_dict)
    assert len(dep_dict) == Config.model.dep_num, 'dep_num should be %d' % len(dep_dict)

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

                    if data == train_data:
                        while True:
                            tree_word_id, tree_pos_id, token_word_id, token_pos_id, buff_top_id, history_action_id, \
                            deque_word_id, deque_pos_id, children_order, stack_order = \
                                parser.extract_from_current_state(sen)
                            legal_transitions = parser.get_legal_transitions(sen)
                            transition = parser.get_oracle_from_current_state(sen)
                            assert legal_transitions[transition] == 1, 'oracle is illegal'

                            example = convert_to_train_example(tree_word_id, tree_pos_id, token_word_id, token_pos_id,
                                                               buff_top_id, history_action_id, deque_word_id,
                                                               deque_pos_id, children_order, stack_order, transition)
                            serialized = example.SerializeToString()
                            tfrecord_writer.write(serialized)
                            j += 1

                            if transition not in [0, 1, 2, 3]:
                                parser.update_tree(sen, transition)  # update composition
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
                        dep_id = [dep2id(t.dep, dep_dict) for t in sen.tokens]
                        example = convert_to_eval_example(word, pos, head, dep_id)
                        serialized = example.SerializeToString()
                        tfrecord_writer.write(serialized)
                        i += 1

                fidx += 1
            print('\n%s complete' % tf_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/tree-lstm+deque.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    Config.data.dataset_path = os.path.expanduser(Config.data.dataset_path)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    create_tfrecord()

    # vocab = load_vocab()
    # tfr = get_tfrecord('test')
    # dataset = tf.data.TFRecordDataset(tfr)
    # dataset = dataset.map(preprocess_eval)
    # dataset = dataset.padded_batch(8, ([-1], [-1], [-1, -1], [-1, -1], []))
    # iterator = dataset.make_one_shot_iterator()
    # next_batch = iterator.get_next('next_batch')
    # sess = tf.Session()
    # r = sess.run(next_batch)
    # print(r[0])
    # print('*'*40)
    # print(r[2])
    # print('*' * 40)
    # print(r[4])
    # print('*'*40)
    # print(r[8])
    # print('*' * 40)
    # print(r[9])
    # print('*' * 40)
    # print(r[11])
