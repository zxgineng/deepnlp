import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import re
import pygame
import cv2

from utils import Config

pygame.init()


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


def build_char_pkl():
    print('creating char pixel pkl ...')
    char_images = np.zeros(
        [int(0x9FA6) - int(0x4E00), Config.model.char_image_size, Config.model.char_image_size, 1])
    font = pygame.font.Font(os.path.join(Config.data.processed_path, "simsun.ttc"), Config.model.char_image_size)
    for n in range(0x4E00, 0x9FA6):
        word = chr(n)
        rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
        img = pygame.surfarray.array3d(rtext)
        img = img.swapaxes(0, 1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img[1:, :]
        img = np.expand_dims(img, -1)
        char_images[n - 19968] = img
    with open(os.path.join(Config.data.processed_path, Config.data.char_pkl), 'wb') as f:
        pickle.dump(char_images, f)


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


def build_tag():
    seg_tag = ['B', 'I', 'E', 'S']
    ner_tag = ['PER', 'LOC', 'ORG']
    tag = [s + '-' + n for s in seg_tag for n in ner_tag] + ['O']
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

    file = os.path.join(Config.data.dataset_path, Config.data.train_data)
    co = re.compile('\t[A-Z]-?[A-Z]*')
    with open(file, encoding='utf8') as f:
        content = re.sub(co, ' ', f.read())
        content = ''.join(content.split())
        vocab.update(content)

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
    sentence = []
    labels = []
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                word, label = line.split()
                sentence.append(word)
                labels.append(label)
            else:
                if set(labels) == {'O'}:
                    continue
                # balance pos and neg data
                new_sentence = []
                new_labels = []
                if ',' in sentence:
                    idx = [0] + [i + 1 for i, s in enumerate(sentence) if s == ','] + [len(sentence)]
                    for i in range(len(idx) - 1):
                        if set(labels[idx[i]:idx[i + 1]]) != {'O'}:
                            new_sentence.extend(sentence[idx[i]:idx[i + 1]])
                            new_labels.extend(labels[idx[i]:idx[i + 1]])
                    if new_sentence[-1] == ',':
                        new_sentence[-1] = '。'
                    sentence = new_sentence
                    labels = new_labels

                total_sentences.append(sentence)
                total_labels.append(labels)
                sentence = []
                labels = []

    return total_sentences, total_labels


def word2image(words, char2image):
    char_image = []
    for char in words:
        if ord(char) >= 19968 and ord(char) <= 40869:
            char_image.append(char2image[ord(char) - 19968])
        else:
            font = pygame.font.Font(os.path.join(Config.data.processed_path, "simsun.ttc"),
                                    Config.model.char_image_size)
            rtext = font.render(char, True, (0, 0, 0), (255, 255, 255))
            img = pygame.surfarray.array3d(rtext)
            img = img.swapaxes(0, 1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img[1:, :]
            if img.shape[0] > img.shape[1]:
                pad_width = (img.shape[0] - img.shape[1]) // 2
                img = np.pad(img, [[0, 0], [pad_width, pad_width]], 'constant', constant_values=(255, 255))
            assert img.shape[0] == img.shape[1]
            img = np.expand_dims(img, -1)
            char_image.append(img)
    return char_image


def word2id(words, vocab):
    word_id = [vocab.get(word, vocab['<UNK>']) for word in words]
    return word_id


def label2id(labels, tag):
    label_id = [tag[label] for label in labels]
    return label_id


def id2label(id, tag):
    id2label = {i: t for i, t in enumerate(tag)}
    return [id2label[i] for i in id]


def convert_to_example(word_id, label_id, char_image):
    """convert one sample to example"""
    char_image = np.array(char_image, np.uint8)
    data = {
        'word_id': _int64_feature(word_id),
        'label_id': _int64_feature(label_id),
        'length': _int64_feature(len(word_id)),
        'char_image': _bytes_feature(char_image.tostring())
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def preprocess(serialized):
    def parse_tfrecord(serialized):
        features = {
            'word_id': tf.VarLenFeature(tf.int64),
            'label_id': tf.VarLenFeature(tf.int64),
            'length': tf.FixedLenFeature([], tf.int64),
            'char_image': tf.FixedLenFeature([], tf.string)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        word_id = tf.sparse_tensor_to_dense(parsed_example['word_id'])
        label_id = tf.sparse_tensor_to_dense(parsed_example['label_id'])
        char_image = tf.decode_raw(parsed_example['char_image'], tf.uint8)
        char_image = tf.reshape(char_image, [-1, Config.model.char_image_size, Config.model.char_image_size, 1])
        length = parsed_example['length']
        return word_id, label_id, char_image, length

    word_id, label_id, char_image, length = parse_tfrecord(serialized)
    char_image = (char_image / 255 - 0.5) * 2
    return word_id, label_id, char_image, length


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
            dataset = dataset.padded_batch(batch_size, (
            [-1], [-1], [-1, Config.model.char_image_size, Config.model.char_image_size, 1], []))

            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            word_id = next_batch[0]
            label_id = next_batch[1]
            char_image = next_batch[2]
            length = next_batch[3]

            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: np.random.permutation(data)})

            return {'word_id': word_id, 'char_image': char_image, 'length': length}, label_id

    return inputs, iterator_initializer_hook


def create_tfrecord():
    build_char_pkl()
    build_vocab()
    build_wordvec_pkl()
    build_tag()
    vocab = load_vocab()
    tag = load_tag()
    char2image = load_char2image()

    if len(tag) != Config.model.fc_unit:
        raise ValueError('length of tag dict must be as same as fc_unit')

    if not os.path.exists(os.path.join(Config.data.processed_path, 'tfrecord')):
        os.makedirs(os.path.join(Config.data.processed_path, 'tfrecord'))

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
            tf_file = os.path.join(Config.data.processed_path, 'tfrecord', tf_file)
            with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
                j = 0
                while i < len(sentences):
                    sys.stdout.write('\r>> converting %s %d/%d' % (data, i + 1, len(sentences)))
                    sys.stdout.flush()
                    word_id = word2id(sentences[i], vocab)
                    label_id = label2id(labels[i], tag)
                    char_images = word2image(sentences[i], char2image)
                    example = convert_to_example(word_id, label_id, char_images)
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
    parser.add_argument('--config', type=str, default='config/cnn-bilstm-crf.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    Config.data.dataset_path = os.path.expanduser(Config.data.dataset_path)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    create_tfrecord()


