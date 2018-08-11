import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import pygame
import cv2

from utils import Config, strQ2B

pygame.init()

PAD = '<PAD>'
UNK = '<UNK>'


def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def _bytes_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(value)))


def pad_to_fixed_len(seq):
    seq = seq[:Config.model.max_seq_length]
    seq = seq + [PAD] * (Config.model.max_seq_length - len(seq))
    return seq


def build_and_read_train(file):
    vocab_file = os.path.join(Config.data.processed_path, Config.data.vocab_file)
    pos_file = os.path.join(Config.data.processed_path, Config.data.pos_file)
    vocab, pos = set(), set()

    all_samples = []
    sample = []

    count = 0
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                line = strQ2B(line)
                content = line.split()
                if count == 0:
                    sample.append(int(content[0]))
                elif count in [2, 5]:
                    sample.append(pad_to_fixed_len(content))
                    vocab.update(content)
                elif count in [3, 6]:
                    sample.append(pad_to_fixed_len(content))
                    pos.update(content)

                count += 1
            else:
                count = 0
                all_samples.append(sample)
                sample = []

        if sample:
            all_samples.append(sample)

    with open(vocab_file, 'w', encoding='utf8') as f:
        f.write('\n'.join([PAD, UNK] + sorted(vocab)))
    with open(pos_file, 'w', encoding='utf8') as f:
        f.write('\n'.join([PAD] + sorted(pos)))

    return all_samples


def read_test(file):
    all_samples = []
    sample = []

    count = 0
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                line = strQ2B(line)
                content = line.split()
                if count == 0:
                    sample.append(int(content[0]))
                elif count in [2, 3, 5, 6]:
                    sample.append(pad_to_fixed_len(content))

                count += 1
            else:
                count = 0
                all_samples.append(sample)
                sample = []

        if sample:
            all_samples.append(sample)

    return all_samples


def load_char2image():
    file = os.path.join(Config.data.processed_path, Config.data.char_pkl)
    with open(file, 'rb') as f:
        char_image = pickle.load(f)
    return char_image


def get_tfrecord(name):
    tfrecords = []
    files = os.listdir(os.path.join(Config.data.processed_path, 'tfrecord/'))
    for file in files:
        if name in file and file.endswith('.tfrecord'):
            tfrecords.append(os.path.join(Config.data.processed_path, 'tfrecord/', file))
    if len(tfrecords) == 0:
        raise RuntimeError('TFrecord not found.')

    return tfrecords


def load_pos():
    tag_file = os.path.join(Config.data.processed_path, Config.data.pos_file)
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


def word2image(words):
    char_images = []
    font = pygame.font.Font(os.path.join(Config.data.processed_path, "SourceHanSerif-Regular.ttc"),
                            Config.model.char_image_size)
    for char in words:
        if len(list(char)) <= Config.model.max_char:
            char_image = []
            for c in list(char):
                try:
                    rtext = font.render(c, True, (0, 0, 0), (255, 255, 255))
                    img = pygame.surfarray.array3d(rtext)
                    img = img.swapaxes(0, 1)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                except Exception as e:
                    print(e.args)
                    img = np.ones((Config.model.char_image_size, Config.model.char_image_size)) * 255
                if img.shape[0] > img.shape[1]:
                    pad_width = (img.shape[0] - img.shape[1]) // 2
                    img = np.pad(img, [[0, 0], [pad_width, pad_width]], 'constant', constant_values=(255, 255))
                img = cv2.resize(img, (Config.model.char_image_size, Config.model.char_image_size))
                char_image.append(img)
            char_image = np.concatenate(char_image, -1)
            width = char_image.shape[1]
            pad_width = int((Config.model.max_char * Config.model.char_image_size - width) / 2)
            char_image = np.pad(char_image, [[0, 0], [pad_width, pad_width]], 'constant', constant_values=(255, 255))
        else:
            try:
                rtext = font.render(char, True, (0, 0, 0), (255, 255, 255))
                img = pygame.surfarray.array3d(rtext)
                img = img.swapaxes(0, 1)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            except Exception as e:
                print(e.args)
                img = np.ones((Config.model.char_image_size, Config.model.char_image_size)) * 255
            char_image = cv2.resize(img, (
                Config.model.max_char * Config.model.char_image_size, Config.model.char_image_size))

        char_image = np.expand_dims(char_image, -1)
        assert char_image.shape == (
            Config.model.char_image_size, Config.model.max_char * Config.model.char_image_size, 1), char_image.shape
        char_images.append(char_image)

    return char_images


def word2id(words, vocab):
    word_id = [vocab.get(word, vocab[UNK]) for word in words]
    return word_id


def pos2id(pos, dict):
    pos_id = [dict[p] for p in pos]
    return pos_id


def convert_to_example(label, p_word_id, p_pos_id, h_word_id, h_pos_id, p_char_images, h_char_images):
    """convert one sample to example"""
    p_char_images = np.array(p_char_images, np.uint8)
    h_char_images = np.array(h_char_images, np.uint8)

    data = {
        'p_word_id': _int64_feature(p_word_id),
        'p_pos_id': _int64_feature(p_pos_id),
        'h_word_id': _int64_feature(h_word_id),
        'h_pos_id': _int64_feature(h_pos_id),
        'label': _int64_feature(label),
        'p_char_images': _bytes_feature(p_char_images.tostring()),
        'h_char_images': _bytes_feature(h_char_images.tostring())
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def preprocess(serialized):
    def parse_tfrecord(serialized):
        features = {
            'p_word_id': tf.FixedLenFeature([Config.model.max_seq_length], tf.int64),
            'p_pos_id': tf.FixedLenFeature([Config.model.max_seq_length], tf.int64),
            'h_word_id': tf.FixedLenFeature([Config.model.max_seq_length], tf.int64),
            'h_pos_id': tf.FixedLenFeature([Config.model.max_seq_length], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'p_char_images': tf.FixedLenFeature([], tf.string),
            'h_char_images': tf.FixedLenFeature([], tf.string)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        p_word_id = parsed_example['p_word_id']
        p_pos_id = parsed_example['p_pos_id']
        h_word_id = parsed_example['h_word_id']
        h_pos_id = parsed_example['h_pos_id']
        label = parsed_example['label']
        p_char_images = tf.decode_raw(parsed_example['p_char_images'], tf.uint8)
        p_char_images = tf.reshape(p_char_images, [Config.model.max_seq_length, Config.model.char_image_size,
                                                   Config.model.max_char * Config.model.char_image_size, 1])
        h_char_images = tf.decode_raw(parsed_example['h_char_images'], tf.uint8)
        h_char_images = tf.reshape(h_char_images, [Config.model.max_seq_length, Config.model.char_image_size,
                                                   Config.model.max_char * Config.model.char_image_size, 1])
        return p_word_id, p_pos_id, h_word_id, h_pos_id, p_char_images, h_char_images, label

    p_word_id, p_pos_id, h_word_id, h_pos_id, p_char_images, h_char_images, label = parse_tfrecord(serialized)
    p_char_images = (p_char_images / 255 - 0.5) * 2
    h_char_images = (h_char_images / 255 - 0.5) * 2
    return p_word_id, p_pos_id, h_word_id, h_pos_id, p_char_images, h_char_images, label


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
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next('next_batch')
        p_word_id = next_batch[0]
        p_pos_id = next_batch[1]
        h_word_id = next_batch[2]
        h_pos_id = next_batch[3]
        p_char_images = next_batch[4]
        h_char_images = next_batch[5]
        label = next_batch[6]

        iterator_initializer_hook.iterator_initializer_func = \
            lambda sess: sess.run(
                iterator.initializer,
                feed_dict={input_placeholder: np.random.permutation(data) if shuffle else data})

        return {'p_word_id': p_word_id, 'p_pos_id': p_pos_id, 'h_word_id': h_word_id, 'h_pos_id': h_pos_id,
                'p_char_images': p_char_images, 'h_char_images': h_char_images}, label

    return inputs, iterator_initializer_hook


def create_tfrecord():
    train_file = os.path.join(Config.data.dataset_path, Config.data.train_data)
    train_data = build_and_read_train(train_file)
    test_file = os.path.join(Config.data.dataset_path, Config.data.test_data)
    test_data = read_test(test_file)
    # build_wordvec_pkl()
    vocab = load_vocab()
    pos_dict = load_pos()

    assert len(pos_dict) == Config.model.pos_num, 'pos_num should be %d' % len(pos_dict)

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
                    label = sample[0]
                    p_word_id = word2id(sample[1], vocab)
                    p_pos_id = pos2id(sample[2], pos_dict)
                    h_word_id = word2id(sample[3], vocab)
                    h_pos_id = pos2id(sample[4], pos_dict)
                    p_char_images = word2image(sample[1])
                    h_char_images = word2image(sample[3])
                    example = convert_to_example(label, p_word_id, p_pos_id, h_word_id, h_pos_id, p_char_images,
                                                 h_char_images)
                    serialized = example.SerializeToString()
                    tfrecord_writer.write(serialized)
                    i += 1
                    j += 1
                    if j >= 1000 and data == train_data:
                        break
                fidx += 1
            print('\n%s complete' % tf_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/diin.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    Config.data.dataset_path = os.path.expanduser(Config.data.dataset_path)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    create_tfrecord()
