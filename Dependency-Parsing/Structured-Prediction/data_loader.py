import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import copy

from utils import Config, strQ2B

NULL = "<NULL>"
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


class Sentence:
    def __init__(self, tokens):
        self.Root = ROOT_TOKEN
        self.tokens = tokens
        self.buff = [token for token in self.tokens]
        self.stack = [self.Root]
        self.dependencies = []
        self.predicted_dependencies = []
        # for beam search
        self.bs_score = 0
        self.bs_action_seq = []
        self.bs_input_seq = []
        self.bs_next_legal_action = None

    def new_branch(self, score, action, inputs):
        new = copy.deepcopy(self)
        new.bs_score = new.bs_score + score
        new.bs_action_seq.append(action)
        new.bs_input_seq.append(inputs)
        return new

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


class ArcStandardParser:
    def __init__(self):
        pass

    def extract_from_stack_and_buffer(self, sentence, num_word=3):
        """extract the last 3 tokens in the stack and the first 3 tokens in the buff, concat them as direct_tokens"""
        tokens = []
        # pad NULL at the beginning until 3 if len(stack) < 3
        tokens.extend([NULL_TOKEN for _ in range(num_word - len(sentence.stack))])
        # extract the last 3 tokens in stack
        tokens.extend(sentence.stack[-num_word:])
        # add the first 3 tokens in buff
        tokens.extend(sentence.buff[:num_word])
        # pad NULL at the end until 3 if len(buff) < 3
        tokens.extend([NULL_TOKEN for _ in range(num_word - len(sentence.buff))])
        # return a list of 6 tokens
        return tokens

    def extract_for_current_state(self, sentence, word_vocab, pos_vocab, dep_vocab):
        """cal direct_tokens and children_tokens to combine current state"""
        direct_tokens = self.extract_from_stack_and_buffer(sentence, Config.data.num_stack_word)  # 6 tokens
        children_tokens = self.extract_children_from_stack(sentence, Config.data.children_stack_range)  # 12 tokens

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

        word_input_ids = [word_vocab.get(word, word_vocab[UNK]) for word in word_features]
        pos_input_ids = [pos_vocab[pos] for pos in pos_features]
        dep_input_ids = [dep_vocab[dep] for dep in dep_features]

        return [word_input_ids, pos_input_ids, dep_input_ids]  # 48 features

    def extract_children_from_stack(self, sentence, num_word=2):
        """extract children from the last 2 token in stack"""
        children_tokens = []
        for i in range(num_word):
            if len(sentence.stack) > i:
                # the first token in the token.left_children
                lc0 = self.get_child_by_index_and_depth(sentence, sentence.stack[-i - 1], 0, "left", 1)
                # the first token in the token.right_children
                rc0 = self.get_child_by_index_and_depth(sentence, sentence.stack[-i - 1], 0, "right", 1)
                # the second token in the token.left_children
                lc1 = self.get_child_by_index_and_depth(sentence, sentence.stack[-i - 1], 1, "left",
                                                        1) if lc0 != NULL_TOKEN else NULL_TOKEN
                # the second token in the token.right_children
                rc1 = self.get_child_by_index_and_depth(sentence, sentence.stack[-i - 1], 1, "right",
                                                        1) if rc0 != NULL_TOKEN else NULL_TOKEN
                # the first token in the left_children of the first token in the token.left_children
                llc0 = self.get_child_by_index_and_depth(sentence, sentence.stack[-i - 1], 0, "left",
                                                         2) if lc0 != NULL_TOKEN else NULL_TOKEN
                # the first token in the right_children of the first token in the token.right_children
                rrc0 = self.get_child_by_index_and_depth(sentence, sentence.stack[-i - 1], 0, "right",
                                                         2) if rc0 != NULL_TOKEN else NULL_TOKEN

                children_tokens.extend([lc0, rc0, lc1, rc1, llc0, rrc0])
            else:
                [children_tokens.append(NULL_TOKEN) for _ in range(6)]
        # return 12 tokens
        return children_tokens

    def get_child_by_index_and_depth(self, sentence, token, index, direction, depth):
        """get child token, return NULL if no child"""
        if depth == 0:
            return token

        if direction == "left":
            if len(token.left_children) > index:
                return self.get_child_by_index_and_depth(sentence,
                                                         sentence.tokens[token.left_children[index] - 1], index,
                                                         direction, depth - 1)
            return NULL_TOKEN
        else:
            if len(token.right_children) > index:
                return self.get_child_by_index_and_depth(sentence,
                                                         sentence.tokens[token.right_children[::-1][index] - 1], index,
                                                         direction, depth - 1)
            return NULL_TOKEN

    def get_legal_labels(self, sentence):
        """check legality of shift, left arc, right arc"""
        labels = [1] if len(sentence.buff) > 0 else [0]
        labels += ([1] if len(sentence.stack) > 2 else [0])
        labels += ([1] if len(sentence.stack) >= 2 else [0])
        labels = [labels[0]] + labels[1:] * (Config.model.dep_num - 2)  # exclude NULL and root in dep dict
        return labels

    def get_transition_from_current_state(self, sentence, dep_dict):
        """get transition according to stack0 and stack1"""
        if len(sentence.stack) < 2:
            return 0  # shift

        stack_token_0 = sentence.stack[-1]
        stack_token_1 = sentence.stack[-2]
        if stack_token_1.token_id >= 1 and stack_token_1.head_id == stack_token_0.token_id:
            return dep_dict[stack_token_1.dep] * 2 + 1  # left arc
        elif stack_token_1.token_id >= 0 and stack_token_0.head_id == stack_token_1.token_id \
                and stack_token_0.token_id not in map(lambda x: x.head_id, sentence.buff):
            return dep_dict[stack_token_0.dep] * 2 + 2  # right arc
        else:
            return 0 if len(sentence.buff) != 0 else None

    def update_state_by_transition(self, sentence, transition, gt=True):
        """updates stack, buffer and dependencies"""
        if transition is not None:
            if transition == 0:  # shift
                sentence.stack.append(sentence.buff[0])
                sentence.buff = sentence.buff[1:] if len(sentence.buff) > 1 else []
            elif transition % 2 == 1:  # left arc
                # save in self.dependencies
                sentence.dependencies.append(
                    (sentence.stack[-1], sentence.stack[-2])) if gt else sentence.predicted_dependencies.append(
                    (sentence.stack[-1], sentence.stack[-2], transition))
                # del the children token
                sentence.stack = sentence.stack[:-2] + sentence.stack[-1:]
            elif transition % 2 == 0:  # right arc
                sentence.dependencies.append(
                    (sentence.stack[-2], sentence.stack[-1])) if gt else sentence.predicted_dependencies.append(
                    (sentence.stack[-2], sentence.stack[-1], transition))
                sentence.stack = sentence.stack[:-1]

    def update_child_dependencies(self, sentence, curr_transition):
        """update left/right children"""
        if curr_transition % 2 == 1:
            head = sentence.stack[-1]
            dependent = sentence.stack[-2]
        elif curr_transition % 2 == 0:
            head = sentence.stack[-2]
            dependent = sentence.stack[-1]

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
                if d != 'root':
                    dep.add(d)
                sen.append(Token(int(i), w, p, d, int(h)))
            else:
                if len(sen) > 1:
                    total_sentences.append(Sentence(sen))
                sen = []

        if len(sen) > 0:
            total_sentences.append(Sentence(sen))

    with open(vocab_file, 'w', encoding='utf8') as f:
        f.write('\n'.join([NULL, UNK, ROOT] + sorted(vocab)))
    with open(pos_file, 'w', encoding='utf8') as f:
        f.write('\n'.join([NULL, ROOT] + sorted(pos)))
    with open(dep_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(sorted(dep) + [NULL, 'root']))

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
                if len(sen) > 1:
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


def id2dep(id, dict):
    id2dep = {i: t for i, t in enumerate(dict)}
    return [id2dep[i] for i in id]


def convert_to_example(idx, word, pos, arc, dep, action_seq):
    """convert one sample to example"""
    data = {
        'idx': _int64_feature(idx),
        'pos': _bytes_feature(pos),
        'arc': _int64_feature(arc),
        'dep': _bytes_feature(dep),
        'word': _bytes_feature(word),
        'action_seq': _int64_feature(action_seq),
        'length': _int64_feature(len(word))
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def preprocess(serialized):
    def parse_tfrecord(serialized):
        features = {
            'idx': tf.VarLenFeature(tf.int64),
            'word': tf.VarLenFeature(tf.string),
            'pos': tf.VarLenFeature(tf.string),
            'arc': tf.VarLenFeature(tf.int64),
            'dep': tf.VarLenFeature(tf.string),
            'action_seq': tf.VarLenFeature(tf.int64),
            'length': tf.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        idx = tf.sparse_tensor_to_dense(parsed_example['idx'])
        word = tf.sparse_tensor_to_dense(parsed_example['word'], default_value='')
        pos = tf.sparse_tensor_to_dense(parsed_example['pos'], default_value='')
        arc = tf.sparse_tensor_to_dense(parsed_example['arc'])
        dep = tf.sparse_tensor_to_dense(parsed_example['dep'], default_value='')
        action_seq = tf.sparse_tensor_to_dense(parsed_example['action_seq'])
        length = parsed_example['length']
        return idx, word, pos, arc, dep, action_seq, length

    return parse_tfrecord(serialized)


def get_dataset_batch(data, buffer_size=1, batch_size=64, scope="train"):
    class IteratorInitializerHook(tf.train.SessionRunHook):

        def __init__(self):
            self.iterator_initializer_func = None

        def after_create_session(self, session, coord):
            self.iterator_initializer_func(session)

    iterator_initializer_hook = IteratorInitializerHook()

    def inputs():
        with tf.name_scope(scope):
            input_placeholder = tf.placeholder(tf.string)
            dataset = tf.data.TFRecordDataset(input_placeholder)
            dataset = dataset.map(preprocess)

            if scope == "train":
                dataset = dataset.repeat(None)  # Infinite iterations
            else:
                dataset = dataset.repeat(1)  # 1 Epoch
            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.padded_batch(batch_size, ([-1], [-1], [-1], [-1], [-1], [-1], []))
            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            idx = next_batch[0]
            word = next_batch[1]
            pos = next_batch[2]
            arc = next_batch[3]
            dep = next_batch[4]
            action_seq = next_batch[5]
            length = next_batch[6]

            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: np.random.permutation(data)})

            return {'idx': idx, 'word': word, 'pos': pos, 'length': length}, \
                   {'action_seq': action_seq, 'arc': arc, 'dep': dep}

    return inputs, iterator_initializer_hook


def create_tfrecord():
    train_file = os.path.join(Config.data.dataset_path, Config.data.train_data)
    train_data = build_and_read_train(train_file)
    test_file = os.path.join(Config.data.dataset_path, Config.data.test_data)
    test_data = read_test(test_file)
    # build_wordvec_pkl()
    pos_dict = load_pos()
    dep_dict = load_dep()

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
                    parser = ArcStandardParser()
                    sen = data[i]
                    action_seq = []
                    idx = [t.token_id for t in sen.tokens]
                    word = [t.word.encode() for t in sen.tokens]
                    pos = [t.pos.encode() for t in sen.tokens]
                    arc = [t.head_id for t in sen.tokens]
                    dep = [t.dep.encode() for t in sen.tokens]

                    num_word = len(sen.tokens)

                    for _ in range(num_word * 2 - 1):
                        legal_labels = parser.get_legal_labels(sen)
                        curr_transition = parser.get_transition_from_current_state(sen, dep_dict)
                        # non-projective
                        if curr_transition is None:
                            print('\nno-projective!!')
                            break
                        assert legal_labels[curr_transition] == 1
                        # update left/right children
                        if curr_transition != 0:
                            parser.update_child_dependencies(sen, curr_transition)
                        # update stack
                        parser.update_state_by_transition(sen, curr_transition)
                        action_seq.append(curr_transition)

                    i += 1
                    if len(action_seq) == num_word * 2 - 1:
                        example = convert_to_example(idx, word, pos, arc, dep, action_seq)
                        serialized = example.SerializeToString()
                        tfrecord_writer.write(serialized)
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

    # tfrecord = get_tfrecord('train')
    # dataset = tf.data.TFRecordDataset(tfrecord)
    # dataset = dataset.map(preprocess)
    # dataset = dataset.padded_batch(2, ([-1], [-1], [-1], [-1], [-1], [-1], []))
    # it = dataset.make_one_shot_iterator()
    # batch = it.get_next()
    # sess = tf.Session()
    # i,w,p,a,d,seq,l = sess.run(batch)
    # print(i)
    # print(w)
    # print(p)
    # print(a)
    # print(d)
    # print(seq)
    # print(l)
