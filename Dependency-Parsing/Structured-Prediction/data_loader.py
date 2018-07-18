import argparse
import os
import sys
import tensorflow as tf
import pickle

from utils import Config

NULL = "<null>"
UNK = "<unk>"
ROOT = "<root>"
pos_prefix = "<p>:"
dep_prefix = "<d>:"
punc_pos = ["''", "``", ":", ".", ","]


def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


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

    def get_transition_from_current_state(self, dep_vocab):
        """get transition according to stack0 and stack1"""
        if len(self.stack) < 2:
            return 0  # shift

        stack_token_0 = self.stack[-1]
        stack_token_1 = self.stack[-2]
        if stack_token_1.token_id >= 1 and stack_token_1.head_id == stack_token_0.token_id:
            return dep_vocab[stack_token_1.dep] * 2 + 1  # left arc
        elif stack_token_1.token_id >= 0 and stack_token_0.head_id == stack_token_1.token_id \
                and stack_token_0.token_id not in map(lambda x: x.head_id, self.buff):
            return dep_vocab[stack_token_0.dep] * 2 + 2  # right arc
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


def convert_to_example(word, pos, dep, label):
    """convert one sample to example"""
    data = {
        'word': _int64_feature(word),
        'pos': _int64_feature(pos),
        'dep': _int64_feature(dep),
        'label': _int64_feature(label)
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def convert_data_to_tfrecord(mode, word_vocab, pos_vocab, dep_vocab):
    file_dict = {'train': 'train.conll', 'val': 'dev.conll'}
    data = read_file(file_dict[mode])
    tf_file = os.path.join(Config.data.base_path, '%s.tfrecord' % (mode))
    with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
        for i, sentence in enumerate(data):
            sys.stdout.write('\r>> converting %s sentence %d/%d' % (mode, i + 1, len(data)))
            sys.stdout.flush()
            num_word = len(sentence.tokens)
            # total num_word*2 steps for one sentence in shift-reduce
            for _ in range(num_word * 2):
                word_input, pos_input, dep_input = extract_for_current_state(sentence, word_vocab, pos_vocab, dep_vocab)
                legal_labels = sentence.get_legal_labels()
                curr_transition = sentence.get_transition_from_current_state(dep_vocab)
                # non-projective
                if curr_transition is None:
                    break
                assert legal_labels[curr_transition] == 1
                # update left/right children
                if curr_transition != 0:
                    sentence.update_child_dependencies(curr_transition)
                # update stack
                sentence.update_state_by_transition(curr_transition)

                example = convert_to_example(word_input, pos_input, dep_input, curr_transition)
                serialized = example.SerializeToString()
                tfrecord_writer.write(serialized)

    print('\n%s sentence convertion completed!' % (mode))


def convert_to_sentence(token_lines, train=True):
    """convert tokens to a sentence object"""
    tokens = []
    for line in token_lines:
        fields = line.strip().split("\t")
        token_index = int(fields[0])
        word = fields[1]
        pos = fields[4]
        dep = fields[7] if train else NULL
        head_index = int(fields[6]) if train else NULL
        token = Token(token_index, word, pos, dep, head_index)
        tokens.append(token)
    sentence = Sentence(tokens)
    return sentence


def extract_for_current_state(sentence, word_vocab, pos_vocab, dep_vocab):
    """cal direct_tokens and children_tokens to combine current state"""
    direct_tokens = extract_from_stack_and_buffer(sentence, Config.data.num_stack_word)  # 6 tokens
    children_tokens = extract_children_from_stack(sentence, Config.data.children_stack_range)  # 12 tokens

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


def extract_children_from_stack(sentence, num_word=2):
    """extract children from the last 2 token in stack"""
    children_tokens = []
    for i in range(num_word):
        if len(sentence.stack) > i:
            # the first token in the token.left_children
            lc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "left", 1)
            # the first token in the token.right_children
            rc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "right", 1)
            # the second token in the token.left_children
            lc1 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 1, "left",
                                                        1) if lc0 != NULL_TOKEN else NULL_TOKEN
            # the second token in the token.right_children
            rc1 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 1, "right",
                                                        1) if rc0 != NULL_TOKEN else NULL_TOKEN
            # the first token in the left_children of the first token in the token.left_children
            llc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "left",
                                                         2) if lc0 != NULL_TOKEN else NULL_TOKEN
            # the first token in the right_children of the first token in the token.right_children
            rrc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "right",
                                                         2) if rc0 != NULL_TOKEN else NULL_TOKEN

            children_tokens.extend([lc0, rc0, lc1, rc1, llc0, rrc0])
        else:
            [children_tokens.append(NULL_TOKEN) for _ in range(6)]
    # return 12 tokens
    return children_tokens


def extract_from_stack_and_buffer(sentence, num_word=3):
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


def load_pkl(file):
    file = os.path.join(Config.data.base_path, file)
    with open(file, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def get_tfrecord(mode):
    return os.path.join(Config.data.base_path, '%s.tfrecord' % (mode))


def get_dataset_batch(data, buffer_size=1000, batch_size=64, scope="train"):
    """create input func"""

    class IteratorInitializerHook(tf.train.SessionRunHook):
        """Hook to initialise data iterator after Session is created."""

        def __init__(self):
            super(IteratorInitializerHook, self).__init__()
            self.iterator_initializer_func = None

        def after_create_session(self, session, coord):
            """Initialise the iterator after the session has been created."""
            self.iterator_initializer_func(session)

    iterator_initializer_hook = IteratorInitializerHook()

    def inputs():
        with tf.name_scope(scope):
            # Define placeholders
            input_placeholder = tf.placeholder(tf.string)
            # Build dataset iterator
            dataset = tf.data.TFRecordDataset(input_placeholder)
            dataset = dataset.map(preprocess)

            if scope == "train":
                dataset = dataset.repeat(Config.train.epoch)
            else:
                dataset = dataset.repeat(1)
            dataset = dataset.shuffle(buffer_size)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            word = next_batch[0]
            pos = next_batch[1]
            dep = next_batch[2]
            label = next_batch[3]

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: data})

            # Return batched (features, labels)
            return {'word': word, 'pos': pos, 'dep': dep}, label

    # Return function and hook
    return inputs, iterator_initializer_hook


def preprocess(serialized):
    def parse_tfrecord(serialized):
        """parse tfrecord"""
        features = {
            'word': tf.FixedLenFeature([18], tf.int64),
            'pos': tf.FixedLenFeature([18], tf.int64),
            'dep': tf.FixedLenFeature([12], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        word = parsed_example['word']
        pos = parsed_example['pos']
        dep = parsed_example['dep']
        label = parsed_example['label']
        return word, pos, dep, label

    return parse_tfrecord(serialized)


def read_file(file, train=True):
    """read file and return a list of sentence object"""
    file = os.path.join(Config.data.base_path, file)
    all_sentences = []
    token_lines = []
    with open(file, encoding='utf8') as f:
        for line in f:
            token_raw = line.strip()
            if len(token_raw) > 0:
                token_lines.append(token_raw)
            else:
                all_sentences.append(convert_to_sentence(token_lines, train=train))
                token_lines = []
        if len(token_lines) > 0:
            all_sentences.append(convert_to_sentence(token_lines))
    return all_sentences


def create_tfrecord():
    word_vocab = load_pkl('word2idx.pkl')
    pos_vocab = load_pkl('pos2idx.pkl')
    dep_vocab = load_pkl('dep2idx.pkl')

    print('writing to tfrecord ...')

    convert_data_to_tfrecord('train', word_vocab, pos_vocab, dep_vocab)
    convert_data_to_tfrecord('val', word_vocab, pos_vocab, dep_vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/mlp-parser.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    create_tfrecord()
