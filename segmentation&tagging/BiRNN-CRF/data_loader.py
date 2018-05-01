import argparse
import os
import sys
import numpy as np
import tensorflow as tf

from utils import Config


seg_labels = ['B', 'M', 'E', 'S']


def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def convert_to_example(ids, tagid):
    """convert one sample to example"""
    data = {
        'id': _int64_feature(ids),
        'tagid': _int64_feature(tagid),
        'length': _int64_feature(len(ids))
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def create_tfrecord():
    """create tfrecord"""
    dataset_fname = os.path.join(Config.data.base_path, Config.data.raw_data)

    build_vocab()
    build_tags(dataset_fname)

    vocab = load_vocab('vocab.txt')
    tags_dict = load_tags('tags.txt')
    if len(tags_dict) != Config.model.fc_unit:
        raise ValueError('length of tags dict must be as same as fc_unit')
    tf_filename = '%s/train.tfrecord' % (Config.data.base_path)
    i = 0

    print('writing to tfrecord ...')
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for contents, tags in zip(Config.data.contents, Config.data.tags):
            sys.stdout.write('\r>> converting sentences %d/%d' % (i + 1, len(Config.data.contents)))
            sys.stdout.flush()

            ids = sentence2id(''.join(contents), vocab)
            tagid = tag2id(contents, tags, tags_dict)

            example = convert_to_example(ids, tagid)
            serialized = example.SerializeToString()
            tfrecord_writer.write(serialized)
            i += 1


def build_vocab():
    """create vocab and tags file"""
    print("create vocab ...")
    words, _ = read_pretrained_wordvec(Config.data.wordvec_fname)
    vocab_fname = os.path.join(Config.data.base_path, 'vocab.txt')
    with open(vocab_fname, mode='w', encoding='utf8') as file:
        file.write('\n'.join(words) + '\n')

def build_tags(fname):
    print('create tags ...')
    contents, tags = [], []
    with open(fname, encoding='utf8') as f:
        for line in f:
            if line.strip():
                c, t = read_sentence(line)
                contents.append(c)
                tags.append(t)
    Config.data.contents = contents
    Config.data.tags = tags
    tag_fname = os.path.join(Config.data.base_path, 'tags.txt')
    with open(tag_fname, mode='w', encoding='utf8') as file:
        file.write('\n'.join(sorted(set(sum(tags, [])))) + '\n')


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
            dataset = dataset.padded_batch(batch_size, ([-1], [-1], [-1], []))

            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            id = next_batch[0]
            segid = next_batch[1]
            tagid = next_batch[2]
            length = next_batch[3]

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: data})

            # Return batched (features, labels)
            return {'id': id, 'length': length}, {'segid':segid, 'tagid':tagid}

    # Return function and hook
    return inputs, iterator_initializer_hook



def load_vocab(vocab_fname):
    """Get the dictionary of tokens and their corresponding index"""
    with open(os.path.join(Config.data.base_path, vocab_fname), encoding='utf8') as f:
        words = f.read().splitlines()
    return {word:i for i,word in enumerate(words)}


def load_tags(tags_fname):
    """Get the dictionary of tags and their corresponding index"""
    total_tags = []
    with open(os.path.join(Config.data.base_path, tags_fname)) as f:
        tags = f.read().splitlines()
    for seg in seg_labels:
        for tag in tags:
            total_tags.append(seg + '-' + tag)
    return {tag:i for i,tag in enumerate(total_tags)}


def preprocess(serialized):
    def parse_tfrecord(serialized):
        """parse tfrecord"""
        features = {
            'id': tf.VarLenFeature(tf.int64),
            'segid': tf.VarLenFeature(tf.int64),
            'tagid': tf.VarLenFeature(tf.int64),
            'length': tf.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        id = tf.sparse_tensor_to_dense(parsed_example['id'])
        segid = tf.sparse_tensor_to_dense(parsed_example['segid'])
        tagid = tf.sparse_tensor_to_dense(parsed_example['tagid'])
        length = parsed_example['length']
        return id, segid, tagid, length

    return parse_tfrecord(serialized)


def read_sentence(sentence):
    contents, tags = [], []
    split = sentence.strip().split(' ')[1:]
    if split:
        for part in split:
            if part:
                if part[0] == '[' and part[1] != '/':
                    p = part[1:].split('/')
                elif ']' in part and ']/' not in part:
                    p = part.split(']')[0].split('/')
                else:
                    p = part.split('/')
                contents.append(p[0])
                tags.append(p[1])
    return contents, tags


def read_pretrained_wordvec(fname):
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
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    id = []
    for token in sentence:
        if token.isdigit():
            id.append(vocab['<NUM>'])
        else:
            id.append(vocab.get(token,vocab['<unk>']))
    return id

def tag2id(tokens, tags, tags_dict):
    """convert tags of a single sentence to an id squence"""
    ids = []
    for token,tag in zip(tokens,tags):
        if len(token) == 1:
            ids.append(tags_dict['S-'+ tag])
        else:
            temp = [tags_dict['M-' + tag]] * len(token)
            temp[0] = tags_dict['B-' + tag]
            temp[-1] = tags_dict['E-' + tag]
            ids.extend(temp)
    return ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/people1998.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    create_tfrecord()

