# import argparse
# import os
# import numpy as np
# import tensorflow as tf
#
# from utils import Config
#
# seg_labels = ['B', 'M', 'E', 'S']
#
#
# def _int64_feature(value):
#     if not isinstance(value, (list, tuple)):
#         return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#     else:
#         return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))
#
#
# def convert_to_example(ids, tagids):
#     """convert one sample to example"""
#     data = {
#         'id': _int64_feature(ids),
#         'tagid': _int64_feature(tagids),
#         'length': _int64_feature(len(ids))
#     }
#     features = tf.train.Features(feature=data)
#     example = tf.train.Example(features=features)
#     return example
#
#
# def create_tfrecord():
#     """create tfrecord"""
#     output_dir = os.path.join(Config.data.base_path, Config.data.processed_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     build_vocab()
#     vocab = load_vocab('vocab.txt')
#     tag_dict = load_tags()
#
#     filenames = sorted(os.listdir(os.path.join(Config.data.base_path, Config.data.raw_path)), reverse=True)
#     tf_filename = '%s/train.tfrecord' % (output_dir)
#     print('writing to tfrecord ...')
#     with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
#         for i, fname in enumerate(filenames):
#             if i % 2 == 0:
#                 sentence = read_raw_text(fname)
#             else:
#                 position, tags = read_labels(fname)
#                 ids = sentence2id(sentence, vocab)
#                 tag_ids = tag2id(sentence, position, tags, tag_dict)
#
#                 example = convert_to_example(ids, tag_ids)
#                 serialized = example.SerializeToString()
#                 tfrecord_writer.write(serialized)
#
#
# def build_vocab():
#     """create vocab"""
#     print("create vocab ...")
#     words, _ = read_pretrained_wordvec(Config.data.wordvec_fname)
#     vocab_fname = os.path.join(Config.data.base_path, Config.data.processed_path, 'vocab.txt')
#     with open(vocab_fname, mode='w', encoding='utf8') as file:
#         file.write('\n'.join(words) + '\n')
#
#
# def get_dataset_batch(data, buffer_size=1000, batch_size=64, scope="train"):
#     """create input func"""
#
#     class IteratorInitializerHook(tf.train.SessionRunHook):
#         """Hook to initialise data iterator after Session is created."""
#
#         def __init__(self):
#             super(IteratorInitializerHook, self).__init__()
#             self.iterator_initializer_func = None
#
#         def after_create_session(self, session, coord):
#             """Initialise the iterator after the session has been created."""
#             self.iterator_initializer_func(session)
#
#     iterator_initializer_hook = IteratorInitializerHook()
#
#     def inputs():
#         with tf.name_scope(scope):
#             # Define placeholders
#             input_placeholder = tf.placeholder(tf.string)
#             # Build dataset iterator
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
#             id = next_batch[0]
#             tagid = next_batch[1]
#             length = next_batch[2]
#
#             # Set runhook to initialize iterator
#             iterator_initializer_hook.iterator_initializer_func = \
#                 lambda sess: sess.run(
#                     iterator.initializer,
#                     feed_dict={input_placeholder: data})
#
#             # Return batched (features, labels)
#             return {'id': id, 'length': length}, {'tagid': tagid}
#
#     # Return function and hook
#     return inputs, iterator_initializer_hook
#
#
# def get_tfrecord():
#     """Get tfrecord file list"""
#     path = os.path.join(Config.data.base_path, Config.data.processed_path)
#     tfr_file = [os.path.join(path, file) for file in os.listdir(path) if 'train' in file]
#     return tfr_file
#
#
# def load_vocab(vocab_fname):
#     """Get the dictionary of tokens and their corresponding index"""
#     with open(os.path.join(Config.data.base_path, Config.data.processed_path, vocab_fname), encoding='utf8') as f:
#         words = f.read().splitlines()
#     return {words[i]: i for i in range(len(words))}
#
#
# def load_tags():
#     """Get the dictionary of tags and their corresponding index"""
#     tags = []
#     for seg in seg_labels:
#         for tag in Config.data.tag_labels:
#             tags.append(seg + '-' + tag)
#     tags.append('O')
#     return {tags[i]: i for i in range(len(tags))}
#
#
# def preprocess(serialized):
#     def parse_tfrecord(serialized):
#         """parse tfrecord"""
#         features = {
#             'id': tf.VarLenFeature(tf.int64),
#             'tagid': tf.VarLenFeature(tf.int64),
#             'length': tf.FixedLenFeature([], tf.int64)
#         }
#         parsed_example = tf.parse_single_example(serialized=serialized, features=features)
#         id = tf.sparse_tensor_to_dense(parsed_example['id'])
#         tagid = tf.sparse_tensor_to_dense(parsed_example['tagid'])
#         length = parsed_example['length']
#         return id, tagid, length
#
#     return parse_tfrecord(serialized)
#
#
# def read_raw_text(fname):
#     """read one sentence from a single file"""
#     path = os.path.join(Config.data.base_path, Config.data.raw_path)
#     with open(os.path.join(path, fname), encoding='utf8') as f:
#         sentence = f.readline().strip()
#     return sentence
#
#
# def read_labels(fname):
#     """read labels from a single file"""
#     path = os.path.join(Config.data.base_path, Config.data.raw_path)
#     position = []
#     tags = []
#     with open(os.path.join(path, fname), encoding='utf8') as f:
#         for line in f:
#             tokens = line.strip().split('\t')
#             if len(tokens) != 4:
#                 continue
#             position.append((int(tokens[1]), int(tokens[2])))
#             tags.append(tokens[3])
#
#     return position, tags
#
#
# def read_pretrained_wordvec(fname):
#     """read pretrained wordvec and convert it to numpy"""
#     fname = os.path.join(Config.data.base_path, fname)
#     words = ['<PAD>']
#     wordvec = [[0.0] * Config.model.embedding_size]
#
#     with open(fname, encoding='utf8') as f:
#         for line in f:
#             split = line.strip().split(' ')
#             word = split[0]
#             vec = split[1:]
#             if len(vec) == Config.model.embedding_size:
#                 words.append(word)
#                 wordvec.append(vec)
#     wordvec = np.array(wordvec)
#
#     return words, wordvec
#
#
# def sentence2id(sentence, vocab):
#     """ Convert all the tokens in the data to their corresponding
#     index in the vocabulary. """
#     ids = []
#     for token in sentence:
#         if token == ' ':
#             ids.append(vocab['<PAD>'])
#         elif token.isdigit():
#             ids.append(vocab['<NUM>'])
#         else:
#             ids.append(vocab.get(token, vocab['<unk>']))
#     return ids
#
#
# def tag2id(sentence, position, tags, tag_dict):
#     """convert tags read from a single file to a id squence"""
#     ids = [tag_dict['O']] * len(sentence)
#     for p, t in zip(position, tags):
#         if p[0] == p[1]:
#             ids[p[0]] = tag_dict['S-' + t]
#         else:
#             ids[p[0]:p[1] + 1] = [tag_dict['M-' + t]] * (p[1] + 1 - p[0])
#             ids[p[0]] = tag_dict['B-' + t]
#             ids[p[1]] = tag_dict['E-' + t]
#     return ids
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--config', type=str, default='config/ner.yml',
#                         help='config file name')
#     args = parser.parse_args()
#
#     Config(args.config)
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


def split_train_test(split_rate=0.05):
    train_files = []
    test_files = []
    dataset_path = os.listdir(Config.data.dataset_path)
    for path in dataset_path:
        path = os.path.join(Config.data.dataset_path, path)
        files = [os.path.join(Config.data.dataset_path, path, file) for file in os.listdir(path)]
        test_f = np.random.choice(files, int(len(files) * split_rate), replace=False)
        test_files.extend(test_f)
        train_f = set(files) - set(test_f)
        train_files.extend(train_f)

    with open(os.path.join(Config.data.processed_path, Config.data.train_data), 'w', encoding='utf8') as f:
        f.write('\n'.join(train_files))
    with open(os.path.join(Config.data.processed_path, Config.data.test_data), 'w', encoding='utf8') as f:
        f.write('\n'.join(test_files))


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
    ner = ['ORG']  # TODO 枚举类型
    tag = [t + '-' + n for t in ['B', 'I', 'S'] for n in ner] + ['O']
    tag_file = os.path.join(Config.data.processed_path, Config.data.tag_file)
    with open(tag_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(tag))


def load_tag():
    tag_file = os.path.join(Config.data.processed_path, Config.data.tag_file)
    with open(tag_file, encoding='utf8') as f:
        tag = f.read().splitlines()
    return {t: i for i, t in enumerate(tag)}


def build_vocab():
    vocab_file = os.path.join(Config.data.processed_path, Config.data.vocab_file)
    vocab = set()

    file = os.path.join(Config.data.processed_path, Config.data.train_data)
    with open(file, encoding='utf8') as f:
        train_files = f.read().split()

    for i, file in enumerate(train_files):
        with open(file, encoding='utf8') as f:
            text = f.read()
            co = re.compile('\s[A-Z]-?[A-Z]*\\n')
            text = re.sub(co, '', text)
            text = ''.join(text.split())
            vocab.update(text)
            sys.stdout.write('\r>> reading text %d/%d' % (i + 1, len(train_files)))
            sys.stdout.flush()
    print()

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


def read_files(files):
    total_sentences, total_labels = [], []
    for i, file in enumerate(files):
        sentences, labels = read_text(file)
        total_sentences.extend(sentences)
        total_labels.extend(labels)
        sys.stdout.write('\r>> reading text into memory %d/%d' % (i + 1, len(files)))
        sys.stdout.flush()
    print()

    return total_sentences, total_labels


def read_text(file):
    sentences = []
    words = []
    labels = []
    la = []
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line and words:
                sentences.append(words)
                labels.append(la)
            else:
                words.append(line.strip()[0])
                la.append(line.strip()[1])
    return sentences, labels


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
    split_train_test()
    build_vocab()
    build_wordvec_pkl()
    vocab = load_vocab()
    tag = load_tag()

    if len(tag) != Config.model.fc_unit:
        raise ValueError('length of tag dict must be as same as fc_unit')

    if not os.path.exists(os.path.join(Config.data.processed_path, 'tfrecord')):
        os.makedirs(os.path.join(Config.data.processed_path, 'tfrecord'))

    print('writing to tfrecord ...')
    for data in [Config.data.train_data, Config.data.test_data]:
        file = os.path.join(Config.data.processed_path, data)
        with open(file, encoding='utf8') as f:
            dataset_files = f.read().split('\n')
        dataset_files = np.random.permutation(dataset_files)

        fidx = 0
        total_i = 0
        for n in range((len(dataset_files) - 1) // 50000 + 1):  # don't have that much memory, data must be split
            part_dataset_files = dataset_files[n * 50000:(n + 1) * 50000]

            sentences, labels = read_files(part_dataset_files)
            i = 0
            while i < len(sentences):
                if data in Config.data.train_data:
                    tf_file = 'tfrecord/train_%d.tfrecord' % fidx
                else:
                    tf_file = 'tfrecord/test.tfrecord'
                tf_file = os.path.join(Config.data.processed_path, tf_file)

                with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
                    j = 0
                    while i < len(sentences):
                        sys.stdout.write('\r>> converting %s %d/%d' % (data, total_i + 1, len(dataset_files)))
                        sys.stdout.flush()
                        word_id = word2id(sentences[i], vocab)
                        label_id = label2id(labels[i], tag)
                        example = convert_to_example(word_id, label_id)
                        serialized = example.SerializeToString()
                        tfrecord_writer.write(serialized)
                        i += 1
                        j += 1
                        total_i += 1
                        if j >= 5000 and data in Config.data.train_data:  # totally shuffled
                            break
                    fidx += 1
                print('\n%s complete' % tf_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/joint-seg-tag.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    Config.data.dataset_path = os.path.expanduser(Config.data.dataset_path)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    create_tfrecord()
