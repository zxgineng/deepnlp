import argparse
import os
import numpy as np
import tensorflow as tf

from utils import Config

seg_labels = ['B', 'M', 'E', 'S']


def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def convert_to_example(ids, tagids):
    """convert one sample to example"""
    data = {
        'id': _int64_feature(ids),
        'tagid': _int64_feature(tagids),
        'length': _int64_feature(len(ids))
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def create_tfrecord():
    """create tfrecord"""
    output_dir = os.path.join(Config.data.base_path, Config.data.processed_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    build_vocab()
    vocab = load_vocab('vocab.txt')
    tag_dict = load_tags()

    filenames = sorted(os.listdir(os.path.join(Config.data.base_path, Config.data.raw_path)), reverse=True)
    tf_filename = '%s/train.tfrecord' % (output_dir)
    print('writing to tfrecord ...')
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, fname in enumerate(filenames):
            if i % 2 == 0:
                sentence = read_raw_text(fname)
            else:
                position, tags = read_labels(fname)
                ids = sentence2id(sentence, vocab)
                tag_ids = tag2id(sentence, position, tags, tag_dict)

                example = convert_to_example(ids, tag_ids)
                serialized = example.SerializeToString()
                tfrecord_writer.write(serialized)


def build_vocab():
    """create vocab"""
    print("create vocab ...")
    words, _ = read_pretrained_wordvec(Config.data.wordvec_fname)
    vocab_fname = os.path.join(Config.data.base_path, Config.data.processed_path, 'vocab.txt')
    with open(vocab_fname, mode='w', encoding='utf8') as file:
        file.write('\n'.join(words) + '\n')


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
                dataset = dataset.repeat(None)  # Infinite iterations
            else:
                dataset = dataset.repeat(1)  # 1 Epoch
            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.padded_batch(batch_size, ([-1], [-1], []))

            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            id = next_batch[0]
            tagid = next_batch[1]
            length = next_batch[2]

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: data})

            # Return batched (features, labels)
            return {'id': id, 'length': length}, {'tagid': tagid}

    # Return function and hook
    return inputs, iterator_initializer_hook


def get_tfrecord():
    """Get tfrecord file list"""
    path = os.path.join(Config.data.base_path, Config.data.processed_path)
    tfr_file = [os.path.join(path, file) for file in os.listdir(path) if 'train' in file]
    return tfr_file


def load_vocab(vocab_fname):
    """Get the dictionary of tokens and their corresponding index"""
    with open(os.path.join(Config.data.base_path, Config.data.processed_path, vocab_fname), encoding='utf8') as f:
        words = f.read().splitlines()
    return {words[i]: i for i in range(len(words))}


def load_tags():
    """Get the dictionary of tags and their corresponding index"""
    tags = []
    for seg in seg_labels:
        for tag in Config.data.tag_labels:
            tags.append(seg + '-' + tag)
    tags.append('O')
    return {tags[i]: i for i in range(len(tags))}


def preprocess(serialized):
    def parse_tfrecord(serialized):
        """parse tfrecord"""
        features = {
            'id': tf.VarLenFeature(tf.int64),
            'tagid': tf.VarLenFeature(tf.int64),
            'length': tf.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        id = tf.sparse_tensor_to_dense(parsed_example['id'])
        tagid = tf.sparse_tensor_to_dense(parsed_example['tagid'])
        length = parsed_example['length']
        return id, tagid, length

    return parse_tfrecord(serialized)


def read_raw_text(fname):
    """read one sentence from a single file"""
    path = os.path.join(Config.data.base_path, Config.data.raw_path)
    with open(os.path.join(path, fname), encoding='utf8') as f:
        sentence = f.readline().strip()
    return sentence


def read_labels(fname):
    """read labels from a single file"""
    path = os.path.join(Config.data.base_path, Config.data.raw_path)
    position = []
    tags = []
    with open(os.path.join(path, fname), encoding='utf8') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) != 4:
                continue
            position.append((int(tokens[1]), int(tokens[2])))
            tags.append(tokens[3])

    return position, tags


def read_pretrained_wordvec(fname):
    """read pretrained wordvec and convert it to numpy"""
    fname = os.path.join(Config.data.base_path, fname)
    words = ['<PAD>']
    wordvec = [[0.0] * Config.model.embedding_size]

    with open(fname, encoding='utf8') as f:
        for line in f:
            split = line.strip().split(' ')
            word = split[0]
            vec = split[1:]
            if len(vec) == Config.model.embedding_size:
                words.append(word)
                wordvec.append(vec)
    wordvec = np.array(wordvec)

    return words, wordvec


def sentence2id(sentence, vocab):
    """ Convert all the tokens in the data to their corresponding
    index in the vocabulary. """
    ids = []
    for token in sentence:
        if token == ' ':
            ids.append(vocab['<PAD>'])
        elif token.isdigit():
            ids.append(vocab['<NUM>'])
        else:
            ids.append(vocab.get(token, vocab['<unk>']))
    return ids


def tag2id(sentence, position, tags, tag_dict):
    """convert tags read from a single file to a id squence"""
    ids = [tag_dict['O']] * len(sentence)
    for p, t in zip(position, tags):
        if p[0] == p[1]:
            ids[p[0]] = tag_dict['S-' + t]
        else:
            ids[p[0]:p[1] + 1] = [tag_dict['M-' + t]] * (p[1] + 1 - p[0])
            ids[p[0]] = tag_dict['B-' + t]
            ids[p[1]] = tag_dict['E-' + t]
    return ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/ner.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    create_tfrecord()
