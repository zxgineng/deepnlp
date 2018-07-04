# import argparse
# import os
# import sys
# import numpy as np
# import tensorflow as tf
# import pickle
#
# from utils import Config
#
#
# def _int64_feature(value):
#     if not isinstance(value, (list, tuple)):
#         return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#     else:
#         return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))
#
# def get_tfrecord(name):
#     tfrecords = []
#     files = os.listdir(Config.data.processed_path)
#     for file in files:
#         if name in file and file.endswith('.tfrecord'):
#             tfrecords.append(os.path.join(Config.data.processed_path, file))
#     if len(tfrecords) == 0:
#         raise RuntimeError('TFrecord not found.')
#
#     return tfrecords
#
#
# def build_tag():
#     ner = ['ORG']  # TODO 枚举类型
#     tag = [t + '-' + n for t in ['B', 'I', 'S'] for n in ner] + ['O']
#     tag_file = os.path.join(Config.data.processed_path, Config.data.tag_file)
#     with open(tag_file, 'w', encoding='utf8') as f:
#         f.write('\n'.join(tag))
#
#
# def load_tag():
#     tag_file = os.path.join(Config.data.processed_path, Config.data.tag_file)
#     with open(tag_file, encoding='utf8') as f:
#         tag = f.read().splitlines()
#     return {t: i for i, t in enumerate(tag)}
#
#
# def build_vocab():
#     vocab_file = os.path.join(Config.data.processed_path, Config.data.vocab_file)
#     vocab = set()
#
#     file = os.path.join(Config.data.processed_path, Config.data.train_data)
#     with open(file, encoding='utf8') as f:
#         train_files = f.read().split()
#
#     for i, file in enumerate(train_files):
#         with open(file, encoding='utf8') as f:
#             text = f.read()
#             co = re.compile('\s[A-Z]-?[A-Z]*\\n')
#             text = re.sub(co, '', text)
#             text = ''.join(text.split())
#             vocab.update(text)
#             sys.stdout.write('\r>> reading text %d/%d' % (i + 1, len(train_files)))
#             sys.stdout.flush()
#     print()
#
#     with open(vocab_file, 'w', encoding='utf8') as f:
#         f.write('\n'.join(['<PAD>', '<UNK>'] + sorted(list(vocab))))
#
#
# def load_vocab():
#     vocab_file = os.path.join(Config.data.processed_path, Config.data.vocab_file)
#     with open(vocab_file, encoding='utf8') as f:
#         words = f.read().splitlines()
#     return {word: i for i, word in enumerate(words)}
#
#
# def build_wordvec_pkl():
#     file = os.path.join(Config.data.processed_path, Config.data.wordvec_file)
#     vocab = load_vocab()
#     vocab_size = len(vocab)
#     wordvec = np.zeros([vocab_size, Config.model.embedding_size])
#
#     with open(file, encoding='utf8') as f:
#         wordvec_dict = {}
#         for line in f:
#             if len(line.split()) < Config.model.embedding_size + 1:
#                 continue
#             word = line.strip().split(' ')[0]
#             vec = line.strip().split(' ')[1:]
#             wordvec_dict[word] = vec
#
#     for index, word in enumerate(vocab):
#         if word in wordvec_dict:
#
#             wordvec[index] = wordvec_dict[word]
#         else:
#             wordvec[index] = np.random.rand(Config.model.embedding_size)
#
#     with open(os.path.join(Config.data.processed_path, Config.data.wordvec_pkl), 'wb') as f:
#         pickle.dump(wordvec, f)
#
#
# def load_pretrained_vec():
#     file = os.path.join(Config.data.processed_path, Config.data.wordvec_pkl)
#     with open(file, 'rb') as f:
#         wordvec = pickle.load(f)
#     return wordvec
#
#
# def read_files(files):
#     total_sentences, total_labels = [], []
#     for i, file in enumerate(files):
#         sentences, labels = read_text(file)
#         total_sentences.extend(sentences)
#         total_labels.extend(labels)
#         sys.stdout.write('\r>> reading text into memory %d/%d' % (i + 1, len(files)))
#         sys.stdout.flush()
#     print()
#
#     return total_sentences, total_labels
#
#
# def read_text(file):
#     sentences = []
#     words = []
#     labels = []
#     la = []
#     with open(file, encoding='utf8') as f:
#         for line in f:
#             line = line.strip()
#             if not line and words:
#                 sentences.append(words)
#                 labels.append(la)
#             else:
#                 words.append(line.strip()[0])
#                 la.append(line.strip()[1])
#     return sentences, labels
#
#
# def word2id(words, vocab):
#     word_id = [vocab.get(word, vocab['<UNK>']) for word in words]
#     return word_id
#
#
# def label2id(labels, tag):
#     label_id = [tag[label] for label in labels]
#     return label_id
#
#
# def id2label(id, tag):
#     id2label = {i: t for i, t in enumerate(tag)}
#     return [id2label[i] for i in id]
#
#
# def convert_to_example(word_id, label_id):
#     """convert one sample to example"""
#     data = {
#         'word_id': _int64_feature(word_id),
#         'label_id': _int64_feature(label_id),
#         'length': _int64_feature(len(word_id))
#     }
#     features = tf.train.Features(feature=data)
#     example = tf.train.Example(features=features)
#     return example
#
#
# def preprocess(serialized):
#     def parse_tfrecord(serialized):
#         features = {
#             'word_id': tf.VarLenFeature(tf.int64),
#             'label_id': tf.VarLenFeature(tf.int64),
#             'length': tf.FixedLenFeature([], tf.int64)
#         }
#         parsed_example = tf.parse_single_example(serialized=serialized, features=features)
#         word_id = tf.sparse_tensor_to_dense(parsed_example['word_id'])
#         label_id = tf.sparse_tensor_to_dense(parsed_example['label_id'])
#         length = parsed_example['length']
#         return word_id, label_id, length
#
#     return parse_tfrecord(serialized)
#
#
# def get_dataset_batch(data, buffer_size=1, batch_size=64, scope="train"):
#     class IteratorInitializerHook(tf.train.SessionRunHook):
#
#         def __init__(self):
#             self.iterator_initializer_func = None
#
#         def after_create_session(self, session, coord):
#             self.iterator_initializer_func(session)
#
#     iterator_initializer_hook = IteratorInitializerHook()
#
#     def inputs():
#         with tf.name_scope(scope):
#             input_placeholder = tf.placeholder(tf.string)
#             dataset = tf.data.TFRecordDataset(input_placeholder)
#             dataset = dataset.map(preprocess)
#
#             if scope == "train":
#                 dataset = dataset.repeat(None)  # Infinite iterations
#             else:
#                 dataset = dataset.repeat(1)  # 1 Epoch
#             dataset = dataset.shuffle(buffer_size=buffer_size)
#             dataset = dataset.padded_batch(batch_size, ([-1], [-1], []))
#
#             iterator = dataset.make_initializable_iterator()
#             next_batch = iterator.get_next()
#             word_id = next_batch[0]
#             label_id = next_batch[1]
#             length = next_batch[2]
#
#             iterator_initializer_hook.iterator_initializer_func = \
#                 lambda sess: sess.run(
#                     iterator.initializer,
#                     feed_dict={input_placeholder: np.random.permutation(data)})
#
#             return {'word_id': word_id, 'length': length}, label_id
#
#     return inputs, iterator_initializer_hook
#
#
# def create_tfrecord():
#     build_vocab()
#     build_wordvec_pkl()
#     vocab = load_vocab()
#     tag = load_tag()
#
#     if len(tag) != Config.model.fc_unit:
#         raise ValueError('length of tag dict must be as same as fc_unit')
#
#     if not os.path.exists(os.path.join(Config.data.processed_path, 'tfrecord')):
#         os.makedirs(os.path.join(Config.data.processed_path, 'tfrecord'))
#
#     print('writing to tfrecord ...')
#     for data in [Config.data.train_data, Config.data.test_data]:
#         file = os.path.join(Config.data.processed_path, data)
#         with open(file, encoding='utf8') as f:
#             dataset_files = f.read().split('\n')
#         dataset_files = np.random.permutation(dataset_files)
#
#         fidx = 0
#         total_i = 0
#         for n in range((len(dataset_files) - 1) // 50000 + 1):  # don't have that much memory, data must be split
#             part_dataset_files = dataset_files[n * 50000:(n + 1) * 50000]
#
#             sentences, labels = read_files(part_dataset_files)
#             i = 0
#             while i < len(sentences):
#                 if data in Config.data.train_data:
#                     tf_file = 'tfrecord/train_%d.tfrecord' % fidx
#                 else:
#                     tf_file = 'tfrecord/test.tfrecord'
#                 tf_file = os.path.join(Config.data.processed_path, tf_file)
#
#                 with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
#                     j = 0
#                     while i < len(sentences):
#                         sys.stdout.write('\r>> converting %s %d/%d' % (data, total_i + 1, len(dataset_files)))
#                         sys.stdout.flush()
#                         word_id = word2id(sentences[i], vocab)
#                         label_id = label2id(labels[i], tag)
#                         example = convert_to_example(word_id, label_id)
#                         serialized = example.SerializeToString()
#                         tfrecord_writer.write(serialized)
#                         i += 1
#                         j += 1
#                         total_i += 1
#                         if j >= 5000 and data in Config.data.train_data:  # totally shuffled
#                             break
#                     fidx += 1
#                 print('\n%s complete' % tf_file)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--config', type=str, default='config/joint-seg-tag.yml',
#                         help='config file name')
#     args = parser.parse_args()
#
#     Config(args.config)
#     Config.data.dataset_path = os.path.expanduser(Config.data.dataset_path)
#     Config.data.processed_path = os.path.expanduser(Config.data.processed_path)
#
#     create_tfrecord()

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import re

from utils import Config


def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def seg_en(text):
    result = []
    temp = ''
    for n in text:
        if str.isalpha(n):
            temp += n
        else:
            if temp:
                result.append(temp)
                temp = ''
            result.append(n)

    if temp:
        result.append(temp)
    return result


def get_tfrecord(name):
    tfrecords = []
    files = os.listdir(Config.data.processed_path)
    for file in files:
        if name in file and file.endswith('.tfrecord'):
            tfrecords.append(os.path.join(Config.data.processed_path, file))
    if len(tfrecords) == 0:
        raise RuntimeError('TFrecord not found.')

    return tfrecords


def build_tag():
    file = os.path.join(Config.data.dataset_path, Config.data.train_data)
    with open(file, encoding='utf8') as f:
        text = f.read().replace('\n', ' ')
        co = re.compile('/[a-zA-Z]+\s')
        pos = set(co.findall(text))

    pos_tag = sorted([p[1:-1] for p in pos])
    seg_tag = ['B', 'M', 'E', 'S']
    tag = [t + '-' + p for t in seg_tag for p in pos_tag]
    tag_file = os.path.join(Config.data.processed_path, Config.data.tag_file)
    with open(tag_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(tag))


def load_tag():
    tag_file = os.path.join(Config.data.processed_path, Config.data.tag_file)
    with open(tag_file, encoding='utf8') as f:
        tag = f.read().splitlines()
    return {t: i for i, t in enumerate(tag)}


def build_vocab():
    if not os.path.exists(Config.data.processed_path):
        os.makedirs(Config.data.processed_path)

    vocab_file = os.path.join(Config.data.processed_path, Config.data.vocab_file)
    vocab = set()

    file = os.path.join(Config.data.dataset_path, Config.data.train_data)
    with open(file, encoding='utf8') as f:
        for line in f:
            if line.split():
                vocab.update(seg_en(line.split()[0]))

    with open(vocab_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(['<PAD>', '<UNK>'] + sorted(list(vocab))))


def load_vocab():
    vocab_file = os.path.join(Config.data.processed_path, Config.data.vocab_file)
    with open(vocab_file, encoding='utf8') as f:
        words = f.read().splitlines()
    return {word: i for i, word in enumerate(words)}


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


def load_pretrained_vec():
    file = os.path.join(Config.data.processed_path, Config.data.wordvec_pkl)
    with open(file, 'rb') as f:
        wordvec = pickle.load(f)
    return wordvec


def read_text(file):
    total_sentences, total_labels = [], []
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                words, labels = parse_sentence(line)
                if not words:
                    continue
                total_sentences.append(words)
                total_labels.append(labels)

    return total_sentences, total_labels


def parse_sentence(sentence):
    labels, words = [], []
    chunks = sentence.split()
    for chunk in chunks:

        word, pos_tag = chunk.split('/')
        assert word and pos_tag
        if not str.isalpha(pos_tag):
            print('skip wrong:', sentence)
            return [], []

        words.append(word)

        if len(word) == 1:
            labels.append('S' + '-' + pos_tag)
        else:
            temp = ['M' + '-' + pos_tag] * len(word)
            temp[0] = 'B' + '-' + pos_tag
            temp[-1] = 'E' + '-' + pos_tag
            labels.extend(temp)
    words = list(''.join(words))
    return words, labels


def word2id(words, vocab):
    word_id = [vocab.get(word, vocab['<UNK>']) for word in words]
    return word_id


def label2id(labels, tag):
    label_id = [tag[label] for label in labels]
    return label_id


def id2label(id, tag):
    id2label = {i: t for i, t in enumerate(tag)}
    return [id2label[i] for i in id]


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
    build_tag()
    vocab = load_vocab()
    tag = load_tag()

    if len(tag) != Config.model.fc_unit:
        raise ValueError('length of tag dict must be as same as fc_unit')

    print('writing to tfrecord ...')
    for data in [Config.data.train_data, Config.data.test_data]:
        dataset_file = os.path.join(Config.data.dataset_path, data)
        sentences, labels = read_text(dataset_file)
        i = 0
        fidx = 0
        while i < len(sentences):
            if data in Config.data.train_data:
                tf_file = 'train_%d.tfrecord' % fidx
            else:
                tf_file = 'test.tfrecord'
            tf_file = os.path.join(Config.data.processed_path, tf_file)
            with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
                j = 0
                while i < len(sentences):
                    sys.stdout.write('\r>> converting %s %d/%d' % (data, i + 1, len(sentences)))
                    sys.stdout.flush()
                    word_id = word2id(sentences[i], vocab)
                    label_id = label2id(labels[i], tag)
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
    parser.add_argument('--config', type=str, default='config/bilstm-crf.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    Config.data.dataset_path = os.path.expanduser(Config.data.dataset_path)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    create_tfrecord()
