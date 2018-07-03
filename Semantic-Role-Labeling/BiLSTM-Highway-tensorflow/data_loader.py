import tensorflow as tf
import pickle
import argparse
import os
import sys

from utils import Config


def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def load_pkl(file):
    file = os.path.join(Config.data.base_path, file)
    with open(file, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def load_dict(file):
    file = os.path.join(Config.data.base_path, file)
    with open(file, 'r') as f:
        vocab = f.read().splitlines()
    return {v: i for i, v in enumerate(vocab)}


def tokens2id(tokens, vocab, lower=True):
    if lower:
        return [vocab.get(token.lower(), '<unk>') for token in tokens]
    else:
        return [vocab.get(token, '<unk>') for token in tokens]


def parse_line(line, word_vocab, tag_vocab):
    parts = line.split('|||')
    sentence = parts[0].strip().split(' ')[1:]
    tags = parts[1].strip()
    word_id = tokens2id(sentence, word_vocab)
    predicate_id = word_vocab.get(sentence[int(line.split(' ')[0])], '<unk>')
    predicate_id = [predicate_id] * len(word_id)
    tag_id = tokens2id(tags.split(' '), tag_vocab, False)
    return word_id, predicate_id, tag_id

def preprocess(serialized):
    def parse_tfrecord(serialized):
        """parse tfrecord"""
        features = {
            'id': tf.VarLenFeature(tf.int64),
            'pid': tf.VarLenFeature(tf.int64),
            'tagid': tf.VarLenFeature(tf.int64),
            'length': tf.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        id = tf.sparse_tensor_to_dense(parsed_example['id'])
        pid = tf.sparse_tensor_to_dense(parsed_example['pid'])
        tagid = tf.sparse_tensor_to_dense(parsed_example['tagid'])
        length = parsed_example['length']
        return id, pid, tagid, length

    return parse_tfrecord(serialized)

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
                dataset = dataset.repeat(1)  # 1 Epoch
            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.padded_batch(batch_size, ([-1], [-1], [-1], []))

            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            id = next_batch[0]
            pid = next_batch[1]
            tagid = next_batch[2]
            length = next_batch[3]

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: data})

            # Return batched (features, labels)
            return {'id': id, 'pid': pid,'length': length},tagid

    # Return function and hook
    return inputs, iterator_initializer_hook


def convert_to_example(word_id, predicate_id, tag_id):
    """convert one sample to example"""
    data = {
        'id': _int64_feature(word_id),
        'pid': _int64_feature(predicate_id),
        'tagid': _int64_feature(tag_id),
        'length': _int64_feature(len(word_id))
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def create_tfrecord():
    word_vocab = load_pkl('word2idx.pkl')
    tag_vocab = load_dict('tags.txt')

    print('writing to tfrecord ...')
    file = os.path.join(Config.data.base_path, 'sample.txt')
    tf_file = '%s/train.tfrecord' % (Config.data.base_path)
    with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
        with open(file) as f:
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                sys.stdout.write('\r>> converting sentences %d/%d' % (i + 1, len(lines)))
                sys.stdout.flush()
                word_id, predicate_id, tag_id = parse_line(line, word_vocab, tag_vocab)
                example = convert_to_example(word_id, predicate_id, tag_id)
                serialized = example.SerializeToString()
                tfrecord_writer.write(serialized)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/sample.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    create_tfrecord()
