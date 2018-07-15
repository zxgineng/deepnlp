import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import pickle

from utils import Config, strQ2B


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
    pos_file = os.path.join(Config.data.processed_path, Config.data.pos_file)
    dep_file = os.path.join(Config.data.processed_path, Config.data.dep_file)
    vocab, pos_tag, dep_tag = set(), set(), set()
    total_sen, total_pos, total_arc, total_dep = [], [], [], []
    sen, pos, arc, dep = ['<ROOT>'], ['<ROOT>'], [], []

    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                line = strQ2B(line)
                _, w, _, p, _, _, a, d, _, _ = line.split()
                vocab.add(w)
                pos_tag.add(p)
                dep_tag.add(d)

                sen.append(w)
                pos.append(p)
                arc.append(int(a))
                dep.append(d)
            else:
                total_sen.append(sen)
                total_pos.append(pos)
                total_arc.append(arc)
                total_dep.append(dep)
                sen, pos, arc, dep = ['<ROOT>'], ['<ROOT>'], [], []

    with open(vocab_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(['<PAD>', '<UNK>', '<ROOT>'] + sorted(vocab)))
    with open(pos_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(['<PAD>', '<ROOT>'] + sorted(pos_tag)))
    with open(dep_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(sorted(dep_tag)))

    return total_sen, total_pos, total_arc, total_dep


def read_test(file):
    total_sen, total_pos, total_arc, total_dep = [], [], [], []
    sen, pos, arc, dep = ['<ROOT>'], ['<ROOT>'], [], []
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                line = strQ2B(line)
                _, w, _, p, _, _, a, d, _, _ = line.split()
                sen.append(w)
                pos.append(p)
                arc.append(int(a))
                dep.append(d)
            else:
                total_sen.append(sen)
                total_pos.append(pos)
                total_arc.append(arc)
                total_dep.append(dep)
                sen, pos, arc, dep = ['<ROOT>'], ['<ROOT>'], [], []

    return total_sen, total_pos, total_arc, total_dep


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


def convert_to_example(word_id, pos_id, arc, dep_id):
    """convert one sample to example"""
    data = {
        'word_id': _int64_feature(word_id),
        'pos_id': _int64_feature(pos_id),
        'arc': _int64_feature(arc),
        'dep_id': _int64_feature(dep_id),
        'length': _int64_feature(len(word_id))
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def preprocess(serialized):
    def parse_tfrecord(serialized):
        features = {
            'word_id': tf.VarLenFeature(tf.int64),
            'pos_id': tf.VarLenFeature(tf.int64),
            'arc': tf.VarLenFeature(tf.int64),
            'dep_id': tf.VarLenFeature(tf.int64),
            'length': tf.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        word_id = tf.sparse_tensor_to_dense(parsed_example['word_id'])
        pos_id = tf.sparse_tensor_to_dense(parsed_example['pos_id'])
        arc = tf.sparse_tensor_to_dense(parsed_example['arc'])
        dep_id = tf.sparse_tensor_to_dense(parsed_example['dep_id'])
        length = parsed_example['length']
        return word_id, pos_id, arc, dep_id, length

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
            dataset = dataset.padded_batch(batch_size, ([-1], [-1], [-1], [-1], []))
            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            word_id = next_batch[0]
            pos_id = next_batch[1]
            arc = next_batch[2]
            dep_id = next_batch[3]
            length = next_batch[4]

            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: np.random.permutation(data)})

            return {'word_id': word_id, 'pos_id': pos_id, 'length': length}, {'arc': arc, 'dep_id': dep_id}

    return inputs, iterator_initializer_hook


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
        sen, pos, arc, dep = data
        i = 0
        fidx = 0
        while i < len(sen):
            if data == train_data:
                tf_file = 'train_%d.tfrecord' % fidx
            else:
                tf_file = 'test.tfrecord'
            tf_file = os.path.join(Config.data.processed_path, 'tfrecord', tf_file)
            with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
                j = 0
                while i < len(sen):
                    sys.stdout.write('\r>> converting %d/%d' % (i + 1, len(sen)))
                    sys.stdout.flush()
                    word_id = word2id(sen[i], vocab)
                    pos_id = pos2id(pos[i], pos_dict)
                    dep_id = dep2id(dep[i], dep_dict)
                    example = convert_to_example(word_id, pos_id, arc[i], dep_id)
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
    parser.add_argument('--config', type=str, default='config/biaffine.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    Config.data.dataset_path = os.path.expanduser(Config.data.dataset_path)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    create_tfrecord()