import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import pickle

from utils import Config

TAGS = {'B': 0, 'M': 1, 'E': 2, 'S': 3}


def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def build_vocab():
    vocab_file = os.path.join(Config.data.processed_path, Config.data.vocab_file)
    vocab = set()
    # add prefix and suffix word
    vocab.update('</>')
    for data in Config.data.train_data:
        vocab.update(data)

    for data in Config.data.train_data:
        file = os.path.join(Config.data.dataset_path, data)
        with open(file, encoding='utf8') as f:
            content = f.read().replace(' ', '').replace('\n', '')
            vocab.update(content)

    with open(vocab_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(['<PAD>', '<UNK>'] + sorted(list(vocab))))


def load_vocab():
    vocab_file = os.path.join(Config.data.processed_path, Config.data.vocab_file)
    with open(vocab_file, encoding='utf8') as f:
        words = f.read().splitlines()
    return {word: i for i, word in enumerate(words)}


def parse_sentence(sentence):
    labels = []
    words = sentence.split()
    for word in words:
        if len(word) == 1:
            labels.append('S')
        else:
            temp = ['M'] * len(word)
            temp[0] = 'B'
            temp[-1] = 'E'
            labels.extend(temp)
    words = list(''.join(words))
    return words, labels


def read_text(file):
    total_sentences, total_labels = [], []
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                words, labels = parse_sentence(line)
                total_sentences.append(words)
                total_labels.append(labels)

    return total_sentences, total_labels


def load_pretrained_vec():
    file = os.path.join(Config.data.processed_path, Config.data.wordvec_pkl)
    with open(file, 'rb') as f:
        wordvec = pickle.load(f)
    return wordvec


def get_tfrecord(name):
    tfrecords = []
    files = os.listdir(Config.data.processed_path)
    for file in files:
        if name in file and file.endswith('.tfrecord'):
            tfrecords.append(os.path.join(Config.data.processed_path, file))
    if len(tfrecords) == 0:
        raise RuntimeError('TFrecord not found.')

    return tfrecords


def id2word(id, vocab):
    id2word = {i: w for i, w in enumerate(vocab)}
    return [id2word[i] for i in id]


def word2id(words, vocab):
    word_id = [vocab.get(word, vocab['<UNK>']) for word in words]
    return word_id


def label2id(labels):
    label_id = [TAGS[label] for label in labels]
    return label_id


def build_wordvec_pkl():
    file = os.path.join(Config.data.processed_path, Config.data.wordvec_file)
    vocab = load_vocab()
    vocab_size = len(vocab)
    wordvec = np.zeros([vocab_size, Config.model.embedding_size])

    with open(file, encoding='utf8') as f:
        wordvec_dict = {}
        for line in f:
            if len(line.split()) < Config.model.embedding_size + 1:
                continue
            word = line.strip().split(' ')[0]
            vec = line.strip().split(' ')[1:]
            wordvec_dict[word] = vec

    for index, word in enumerate(vocab):
        if word in wordvec_dict:

            wordvec[index] = wordvec_dict[word]
        else:
            wordvec[index] = np.random.rand(Config.model.embedding_size)

    with open(os.path.join(Config.data.processed_path, Config.data.wordvec_pkl), 'wb') as f:
        pickle.dump(wordvec, f)


def convert_to_example(word_id, label_id):
    """convert one sample to example"""
    data = {
        'word_id': _int64_feature(word_id),
        'label_id': _int64_feature(label_id),
        'length': _int64_feature(len(word_id))
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def preprocess(serialized):
    def parse_tfrecord(serialized):
        features = {
            'word_id': tf.VarLenFeature(tf.int64),
            'label_id': tf.VarLenFeature(tf.int64),
            'length': tf.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        word_id = tf.sparse_tensor_to_dense(parsed_example['word_id'])
        label_id = tf.sparse_tensor_to_dense(parsed_example['label_id'])
        length = parsed_example['length']
        return word_id, label_id, length

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
            dataset = dataset.padded_batch(batch_size, ([-1], [-1], []))

            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            word_id = next_batch[0]
            label_id = next_batch[1]
            length = next_batch[2]

            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: np.random.permutation(data)})

            return {'word_id': word_id, 'length': length}, label_id

    return inputs, iterator_initializer_hook


def create_tfrecord():
    build_vocab()
    build_wordvec_pkl()
    vocab = load_vocab()

    if len(TAGS) != Config.model.fc_unit:
        raise ValueError('length of tag dict must be as same as fc_unit')

    print('writing to tfrecord ...')
    for data in Config.data.train_data + Config.data.test_data:
        word_prefix = list('<%s>' % data.split('_')[0])
        word_suffix = list('</%s>' % data.split('_')[0])
        label_prefix = ['B'] + (len(word_prefix) - 2) * ['M'] + ['E']
        label_suffix = ['B'] + (len(word_suffix) - 2) * ['M'] + ['E']

        dataset_file = os.path.join(Config.data.dataset_path, data)
        sentences, labels = read_text(dataset_file)
        i = 0
        fidx = 0
        while i < len(sentences):
            if data in Config.data.train_data:
                tf_file = '%s_train_%d.tfrecord' % (data.split('_')[0], fidx)
            else:
                tf_file = '%s_test.tfrecord' % (data.split('_')[0])
            tf_file = os.path.join(Config.data.processed_path, tf_file)
            with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
                j = 0
                while i < len(sentences):
                    s = word_prefix + sentences[i] + word_suffix
                    l = label_prefix + labels[i] + label_suffix
                    sys.stdout.write('\r>> converting %s %d/%d' % (data, i + 1, len(sentences)))
                    sys.stdout.flush()
                    word_id = word2id(s, vocab)
                    label_id = label2id(l)
                    example = convert_to_example(word_id, label_id)
                    serialized = example.SerializeToString()
                    tfrecord_writer.write(serialized)
                    i += 1
                    j += 1
                    if j >= 5000 and data in Config.data.train_data:  # totally shuffled
                        break
                fidx += 1
            print('\n%s complete' % tf_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/multi-criteria.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    Config.data.dataset_path = os.path.expanduser(Config.data.dataset_path)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    create_tfrecord()
