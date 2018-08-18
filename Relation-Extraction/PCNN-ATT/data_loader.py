import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import pickle

from utils import Config, strQ2B

PAD = '<PAD>'
UNK = '<UNK>'


def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def pos_encode(pos_list):
    outputs = []
    for pos in pos_list:
        outputs.append(max(1, min(100 + pos, 200)))
    return outputs


def build_and_read_train(file):
    vocab_file = os.path.join(Config.data.processed_path, Config.data.vocab_file)
    rel_file = os.path.join(Config.data.processed_path, Config.data.rel_file)
    entity_file = os.path.join(Config.data.processed_path, Config.data.entity_file)
    vocab, rel, entity_pairs = set(), set(), set()

    all_samples = []

    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                line = strQ2B(line)
                en1, en2, r, sen = line.split('\t')
                entity_pairs.add(en1 + ' ' + en2)
                vocab.update(sen)
                rel.add(r)
                all_samples.append((en1, en2, r, sen))

    with open(vocab_file, 'w', encoding='utf8') as f:
        f.write('\n'.join([PAD, UNK] + sorted(vocab)))
    with open(rel_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(sorted(rel)))
    with open(entity_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(sorted(entity_pairs)))
    return all_samples


def get_tfrecord(name):
    tfrecords = []
    files = os.listdir(os.path.join(Config.data.processed_path, 'tfrecord/'))
    for file in files:
        if name in file and file.endswith('.tfrecord'):
            tfrecords.append(os.path.join(Config.data.processed_path, 'tfrecord/', file))
    if len(tfrecords) == 0:
        raise RuntimeError('TFrecord not found.')

    return tfrecords


def load_entity_pair():
    entity_file = os.path.join(Config.data.processed_path, Config.data.entity_file)
    with open(entity_file, encoding='utf8') as f:
        entity_pair = f.read().splitlines()
    return {t: i for i, t in enumerate(entity_pair)}


def load_rel():
    rel_file = os.path.join(Config.data.processed_path, Config.data.rel_file)
    with open(rel_file, encoding='utf8') as f:
        rel = f.read().splitlines()
    return {r: i for i, r in enumerate(rel)}


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

def id2rel(id, dict):
    id2rel = {i: t for i, t in enumerate(dict)}
    return [id2rel[i] for i in id]

def convert_to_example(entity_pair_id, label, word_id, pos_1, pos_2, en1_pos, en2_pos):
    """convert one sample to example"""

    data = {
        'entity_pair_id': _int64_feature(entity_pair_id),
        'label': _int64_feature(label),
        'word_id': _int64_feature(word_id),
        'pos_1': _int64_feature(pos_1),
        'pos_2': _int64_feature(pos_2),
        'en1_pos': _int64_feature(en1_pos),
        'en2_pos': _int64_feature(en2_pos)
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def preprocess(serialized):
    def parse_tfrecord(serialized):
        features = {
            'entity_pair_id': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'word_id': tf.VarLenFeature(tf.int64),
            'pos_1': tf.VarLenFeature(tf.int64),
            'pos_2': tf.VarLenFeature(tf.int64),
            'en1_pos': tf.FixedLenFeature([], tf.int64),
            'en2_pos': tf.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        entity_pair_id = parsed_example['entity_pair_id']
        label = parsed_example['label']
        word_id = tf.sparse_tensor_to_dense(parsed_example['word_id'])
        pos_1 = tf.sparse_tensor_to_dense(parsed_example['pos_1'])
        pos_2 = tf.sparse_tensor_to_dense(parsed_example['pos_2'])
        en1_pos = parsed_example['en1_pos']
        en2_pos = parsed_example['en2_pos']

        return entity_pair_id, label, word_id, pos_1, pos_2, en1_pos, en2_pos

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
            dataset = dataset.repeat(1)  # 1 Epoch
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.padded_batch(batch_size, ([], [], [-1], [-1], [-1], [], []))

        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next('next_batch')
        entity_pair_id = next_batch[0]
        label = next_batch[1]
        word_id = next_batch[2]
        pos_1 = next_batch[3]
        pos_2 = next_batch[4]
        en1_pos = next_batch[5]
        en2_pos = next_batch[6]

        iterator_initializer_hook.iterator_initializer_func = \
            lambda sess: sess.run(
                iterator.initializer,
                feed_dict={input_placeholder: np.random.permutation(data) if shuffle else data})

        return {'word_id': word_id, 'pos_1': pos_1, 'pos_2': pos_2,
                'en1_pos': en1_pos, 'en2_pos': en2_pos}, {'entity_pair_id': entity_pair_id, 'label': label}

    return inputs, iterator_initializer_hook


def create_tfrecord():
    train_file = os.path.join(Config.data.dataset_path, Config.data.train_data)
    train_data = build_and_read_train(train_file)
    build_wordvec_pkl()
    vocab = load_vocab()
    entity_pair_dict = load_entity_pair()
    rel_dict = load_rel()

    assert len(rel_dict) == Config.model.class_num, 'class_num should be %d' % len(rel_dict)
    if not os.path.exists(os.path.join(Config.data.processed_path, 'tfrecord')):
        os.makedirs(os.path.join(Config.data.processed_path, 'tfrecord'))

    print('writing to tfrecord ...')
    for data in [train_data]:
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
                    en1, en2, r, sen = data[i]
                    if sen.count(en1) < 1 or sen.count(en2) < 1:
                        i += 1
                        continue
                    entity_pair_id = entity_pair_dict[en1 + ' ' + en2]
                    en1_start = sen.index(en1)
                    en1_end = en1_start + len(en1) -1
                    en2_start = sen.index(en2)
                    en2_end = en2_start + len(en2) - 1
                    label = rel_dict[r]
                    word_id = word2id(list(sen), vocab)
                    pos_1 = []
                    pos_2 = []
                    for n in range(len(sen)):
                        if n < en1_start:
                            pos_1.append(n - en1_start)
                        elif en1_start <= n <= en1_end:
                            pos_1.append(0)
                        else:
                            pos_1.append(n - en1_end)

                        if n < en2_start:
                            pos_2.append(n - en2_start)
                        elif en2_start <= n <= en2_end:
                            pos_2.append(0)
                        else:
                            pos_2.append(n - en2_end)

                    example = convert_to_example(entity_pair_id, label, word_id, pos_encode(pos_1), pos_encode(pos_2),
                                                 en1_end, en2_end)
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
    parser.add_argument('--config', type=str, default='config/pcnn-att.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    Config.data.dataset_path = os.path.expanduser(Config.data.dataset_path)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    create_tfrecord()
