import argparse
import os
import sys
from collections import Counter
import numpy as np
import tensorflow as tf

from utils import Config


def precess_data():
    print('Preparing data to be model-ready ...')

    path = os.path.join(Config.data.base_path, Config.data.processed_path)
    if not os.path.exists(path):
        os.mkdir(path)

    build_vocab('cnews.train.txt')
    token2id('cnews.train.txt', 'train')
    token2id('cnews.test.txt', 'test')
    token2id('cnews.val.txt', 'val')


def _convert_category(categories):
    """label to index"""
    categories = sorted(set(categories))
    cat_dict = dict(zip(categories, range(len(categories))))
    return cat_dict


def read_file(filename):
    """read content and label from file"""
    contents, labels = [], []

    with open(filename, encoding='utf8') as f:
        for line in f:
            label, content = line.strip().split('\t')
            if content:
                contents.append(list(content))
                labels.append(label)

    return contents, labels


def get_dataset_batch(data, buffer_size=5000, batch_size=64, scope="train"):
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

            X, y = data

            # Define placeholders
            input_placeholder = tf.placeholder(
                tf.int32, [None, Config.data.max_seq_length - 1])
            output_placeholder = tf.placeholder(
                tf.int32, [None])

            # Build dataset iterator
            dataset = tf.data.Dataset.from_tensor_slices(
                (input_placeholder, output_placeholder))

            if scope == "train":
                dataset = dataset.repeat(None)  # Infinite iterations
            else:
                dataset = dataset.repeat(1)  # 1 Epoch
            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            next_X, next_y = iterator.get_next()

            tf.identity(next_X[0], 'input_0')
            tf.identity(next_y[0], 'target_0')

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: X,
                               output_placeholder: y})

            # Return batched (features, labels)
            return {'inputs': next_X}, next_y

    # Return function and hook
    return inputs, iterator_initializer_hook


def build_vocab(fname):
    """Count the vocab and write into the file"""
    print("Count each vocab frequency ...")

    path = os.path.join(Config.data.base_path, Config.data.raw_data_path, fname)
    contents, labels = read_file(path)

    cat_dict = _convert_category(labels)
    Config.cat_dict = cat_dict
    assert Config.data.num_classes == len(cat_dict.keys())
    print('total classes: %s' % (', '.join(cat_dict.keys())))

    all_data = []
    for content in contents:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(Config.data.vocab_size - 2)
    words, _ = list(zip(*count_pairs))

    words = ['<PAD>', '<UNK>'] + list(words)
    vocab_path = os.path.join(Config.data.base_path, Config.data.processed_path, 'vocab.txt')
    with open(vocab_path, mode='w', encoding='utf8') as file:
        file.write('\n'.join(words) + '\n')


def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<UNK>']) for token in line]


def token2id(fname, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab.txt'
    out_path = mode + '_ids.txt'

    vocab = load_vocab(vocab_path)
    contents, labels = read_file(os.path.join(Config.data.base_path, Config.data.raw_data_path, fname))
    out_file = open(os.path.join(Config.data.base_path, Config.data.processed_path, out_path), 'wb')

    # covert into index
    for i, sentence in enumerate(contents):
        ids = [Config.cat_dict[labels[i]]]
        sentence_ids = sentence2id(vocab, sentence)
        ids.extend(sentence_ids)
        out_file.write(b' '.join(str(id_).encode('utf-8') for id_ in ids) + b'\n')
        sys.stdout.write('\r>> %s sentences to idx %d/%d' % (mode, i + 1, len(contents)))
        sys.stdout.flush()
    print()


def load_vocab(vocab_fname):
    """Get the dictionary of tokens and their corresponding index"""
    print("load vocab ...")
    with open(os.path.join(Config.data.base_path, Config.data.processed_path, vocab_fname), 'rb') as f:
        words = f.read().decode('utf-8').splitlines()
        print("vocab size:", len(words))
    return {words[i]: i for i in range(len(words))}


def pad_input(input_, size):
    return input_ + ['0'] * (size - len(input_))


def load_data(fname):
    data_file = open(os.path.join(Config.data.base_path, Config.data.processed_path, fname), 'r')

    data = []
    for line in data_file.readlines():
        sentence = line.split()
        if len(sentence) < Config.data.max_seq_length:
            data.append(pad_input(sentence, Config.data.max_seq_length))
        else:
            data.append(sentence[:Config.data.max_seq_length])

    data = np.array(data, dtype=np.int32)
    return data[:, 1:], data[:, 0]


def make_data_set(mode):
    if Config.data.get('max_seq_length', None) is None:
        set_max_seq_length(['train_ids.txt', 'val_ids.txt'])

    if mode == 'train':
        train_features, train_labels = load_data('train_ids.txt')
        print(f"train data count : {len(train_features)}")
        return (train_features, train_labels)

    elif mode == 'train_and_val':
        train_features, train_labels = load_data('train_ids.txt')
        val_features, val_labels = load_data('val_ids.txt')
        print(f"train data count : {len(train_features)}")
        print(f"eval data count : {len(val_features)}")
        return (train_features, train_labels), (val_features[:100], val_labels[:100])

    elif mode == 'test':
        test_features, test_labels = load_data('test_ids.txt')
        print(f"test data count : {len(test_features)}")
        return (test_features, test_labels)


    else:
        raise ValueError('no %s mode!' % (mode))


def set_max_seq_length(dataset_fnames):
    """
    Count the max seq length of the file.
    Save into Config
    :param dataset_fnames: list of str, processed ids files
    """
    max_seq_length = 10

    for fname in dataset_fnames:
        input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, fname), 'r')
        # count the max seq length of the file
        for line in input_data.readlines():
            seq_length = len(line.split())
            if seq_length > max_seq_length:
                max_seq_length = seq_length

    Config.data.max_seq_length = max_seq_length
    print(f"Setting max_seq_length to Config : {max_seq_length}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/thunews.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    precess_data()
