import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import pickle

from utils import Config, strQ2B

UNK = "<UNK>"
PAD = '<PAD>'


def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


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
    tag_file = os.path.join(Config.data.processed_path, Config.data.tag_file)
    vocab, tag = set(), set()
    sen = []
    total_sen = []

    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                line = strQ2B(line)
                w, t, p, l = line.split()
                vocab.add(w)
                tag.add(t)
                sen.append([w, t, int(p), l])
            else:
                total_sen.append(sen)
                sen = []

        if sen:
            total_sen.append(sen)

    with open(vocab_file, 'w', encoding='utf8') as f:
        f.write('\n'.join([PAD, UNK] + sorted(vocab)))
    with open(tag_file, 'w', encoding='utf8') as f:
        f.write('\n'.join([PAD] + sorted(tag)))

    return total_sen


def read_test(file):
    sen = []
    total_sen = []
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                line = strQ2B(line)
                w, t, p, l = line.split()
                sen.append([w, t, int(p), l])
            else:
                total_sen.append(sen)
                sen = []

        if sen:
            total_sen.append(sen)

    return total_sen


def build_label():
    label_file = os.path.join(Config.data.processed_path, Config.data.label_file)
    prefix = ['B', 'I', 'E', 'S']
    label = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGM-ADV', 'ARGM-BNF', 'ARGM-CND', 'ARGM-DIR', 'ARGM-DIS',
             'ARGM-DGR', 'ARGM-EXT', 'ARGM-FRQ', 'ARGM-LOC', 'ARGM-MNR', 'ARGM-PRP', 'ARGM-TMP', 'ARGM-TPC']
    total_labels = ['O', 'rel'] + [p + '-' + l for l in label for p in prefix]
    with open(label_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(total_labels))


def load_tag():
    tag_file = os.path.join(Config.data.processed_path, Config.data.tag_file)
    with open(tag_file, encoding='utf8') as f:
        tag = f.read().splitlines()
    return {t: i for i, t in enumerate(tag)}


def load_label():
    label_file = os.path.join(Config.data.processed_path, Config.data.label_file)
    with open(label_file, encoding='utf8') as f:
        label = f.read().splitlines()
    return {l: i for i, l in enumerate(label)}


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


def tag2id(tag, dict):
    tag_id = [dict[t] for t in tag]
    return tag_id


def label2id(label, dict):
    label_id = [dict[l] for l in label]
    return label_id


def id2label(id, dict):
    id2label = {i: l for i, l in enumerate(dict)}
    return [id2label[i] for i in id]


def convert_to_example(word_id, tag_id, predicate, label):
    """convert one sample to example"""
    data = {
        'word_id': _int64_feature(word_id),
        'tag_id': _int64_feature(tag_id),
        'predicate': _int64_feature(predicate),
        'label': _int64_feature(label),
        'length': _int64_feature(len(word_id))
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def preprocess(serialized):
    def parse_tfrecord(serialized):
        features = {
            'word_id': tf.VarLenFeature(tf.int64),
            'tag_id': tf.VarLenFeature(tf.int64),
            'predicate': tf.VarLenFeature(tf.int64),
            'label': tf.VarLenFeature(tf.int64),
            'length': tf.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        word_id = tf.sparse_tensor_to_dense(parsed_example['word_id'])
        tag_id = tf.sparse_tensor_to_dense(parsed_example['tag_id'])
        predicate = tf.sparse_tensor_to_dense(parsed_example['predicate'])
        label = tf.sparse_tensor_to_dense(parsed_example['label'])
        length = parsed_example['length']
        return word_id, tag_id, predicate, length, label

    return parse_tfrecord(serialized)


def get_dataset_batch(data, buffer_size=1, batch_size=64, scope="train", shuffle=True):
    class IteratorInitializerHook(tf.train.SessionRunHook):

        def __init__(self):
            self.iterator_initializer_func = None

        def after_create_session(self, session, coord):
            self.iterator_initializer_func(session)

    iterator_initializer_hook = IteratorInitializerHook()

    def inputs():
        input_placeholder = tf.placeholder(tf.string)
        dataset = tf.data.TFRecordDataset(input_placeholder)
        dataset = dataset.map(preprocess)

        if scope == "train":
            dataset = dataset.repeat(None)  # Infinite iterations
        else:
            dataset = dataset.repeat(5)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.padded_batch(batch_size, ([-1], [-1], [-1], [], [-1]))
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next('next_batch')
        word_id = next_batch[0]
        tag_id = next_batch[1]
        predicate = next_batch[2]
        length = next_batch[3]
        label = next_batch[4]

        iterator_initializer_hook.iterator_initializer_func = \
            lambda sess: sess.run(
                iterator.initializer,
                feed_dict={input_placeholder: np.random.permutation(data) if shuffle else data})

        return {'word_id': word_id, 'tag_id': tag_id, 'predicate': predicate, 'length': length}, label

    return inputs, iterator_initializer_hook


def create_tfrecord():
    train_file = os.path.join(Config.data.dataset_path, Config.data.train_data)
    train_data = build_and_read_train(train_file)
    test_file = os.path.join(Config.data.dataset_path, Config.data.test_data)
    test_data = read_test(test_file)
    build_label()
    build_wordvec_pkl()
    vocab = load_vocab()
    tag_dict = load_tag()
    label_dict = load_label()

    assert len(tag_dict) == Config.model.tag_num, 'tag_num should be %d' % len(tag_dict)
    assert len(label_dict) == Config.model.class_num, 'class_num should be %d' % len(label_dict)

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
                    sample = data[i]
                    word, tag, pre, label = [], [], [], []
                    for n in sample:
                        word.append(n[0])
                        tag.append(n[1])
                        pre.append(n[2])
                        label.append(n[3])
                    word_id = word2id(word, vocab)
                    tag_id = tag2id(tag, tag_dict)
                    label = label2id(label, label_dict)

                    example = convert_to_example(word_id, tag_id, pre, label)
                    serialized = example.SerializeToString()
                    tfrecord_writer.write(serialized)
                    i += 1
                    j += 1
                    if j >= 5000 and data == train_data:
                        break
                fidx += 1
            print('\n%s complete' % tf_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/bilstm-highway.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    Config.data.dataset_path = os.path.expanduser(Config.data.dataset_path)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    create_tfrecord()

