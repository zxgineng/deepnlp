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


def split_train_test(split_rate=0.03):
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


def get_tfrecord(folder, name):
    tfrecords = []
    files = os.listdir(os.path.join(Config.data.processed_path, folder))
    for file in files:
        if name in file and file.endswith('.tfrecord'):
            tfrecords.append(os.path.join(Config.data.processed_path, folder, file))
    if len(tfrecords) == 0:
        raise RuntimeError('TFrecord not found.')

    return tfrecords


def build_vocab():
    vocab_file = os.path.join(Config.data.processed_path, Config.data.vocab_file)
    vocab = set()

    file = os.path.join(Config.data.processed_path, Config.data.train_data)
    with open(file, encoding='utf8') as f:
        train_files = f.read().split()

    for i, file in enumerate(train_files):
        with open(file, encoding='utf8') as f:
            content = strQ2B(''.join(f.read().split()))
            vocab.update(content)
            sys.stdout.write('\r>> reading text %d/%d' % (i + 1, len(train_files)))
            sys.stdout.flush()
    print()

    with open(vocab_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(['<UNK>', '<START/>', '</END>'] + sorted(list(vocab))))


def load_vocab():
    vocab_file = os.path.join(Config.data.processed_path, Config.data.vocab_file)
    with open(vocab_file, encoding='utf8') as f:
        words = f.read().splitlines()
    assert len(words) == Config.model.vocab_num
    return {word: i for i, word in enumerate(words)}


def build_char_pkl():
    print('creating char pixel pkl ...')
    vocab = load_vocab()
    char_images = {}
    font = pygame.font.Font(os.path.join(Config.data.processed_path, "SourceHanSerif-Regular.ttc"),
                            Config.model.char_image_size)
    for char in vocab:
        if char == '<UNK>':
            continue
        try:
            rtext = font.render(char, True, (0, 0, 0), (255, 255, 255))
            img = pygame.surfarray.array3d(rtext)
            img = img.swapaxes(0, 1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img[1:, :]
        except Exception as e:
            print(e.args)
            img = np.ones((Config.model.char_image_size, Config.model.char_image_size)) * 255
        if img.shape[0] > img.shape[1]:
            pad_width = (img.shape[0] - img.shape[1]) // 2
            img = np.pad(img, [[0, 0], [pad_width, pad_width]], 'constant', constant_values=(255, 255))
        img = cv2.resize(img, (Config.model.char_image_size, Config.model.char_image_size))
        img = np.expand_dims(img, -1)
        char_images[char] = img

    with open(os.path.join(Config.data.processed_path, Config.data.char_pkl), 'wb') as f:
        pickle.dump(char_images, f)


def load_char2image():
    file = os.path.join(Config.data.processed_path, Config.data.char_pkl)
    with open(file, 'rb') as f:
        char_image = pickle.load(f)
    return char_image


def read_file(file):
    with open(file, encoding='utf8') as f:
        content = f.read().replace('。', '。\n').replace('?', '?\n').replace('!', '!\n').replace('……', '……\n').replace(
            '\u3000', ' ')
        sentences = content.split('\n')

        for_samples = []
        for_labels = []
        inputs_temp = []
        labels_temp = []
        for sen in sentences:
            sen = sen.strip()
            if sen:
                inputs = ['<START/>'] + list(sen)
                labels = list(sen) + ['</END>']
                l = Config.model.seq_length - len(inputs_temp)
                inputs_temp.extend(inputs[:l])
                labels_temp.extend(labels[:l])

            if len(inputs_temp) == Config.model.seq_length:
                for_samples.append(inputs_temp)
                for_labels.append(labels_temp)
                inputs_temp = []
                labels_temp = []

        back_samples = []
        back_labels = []
        inputs_temp = []
        labels_temp = []
        for sen in reversed(sentences):
            sen = sen.strip()
            if sen:
                inputs = ['</END>'] + list(reversed(sen))
                labels = list(reversed(sen)) + ['<START/>']
                l = Config.model.seq_length - len(inputs_temp)
                inputs_temp.extend(inputs[:l])
                labels_temp.extend(labels[:l])

            if len(inputs_temp) == Config.model.seq_length:
                back_samples.append(inputs_temp)
                back_labels.append(labels_temp)
                inputs_temp = []
                labels_temp = []

    return for_samples, for_labels, back_samples, back_labels


def word2image(words, char2image):
    char_images = []
    for char in words:
        if char in char2image:
            char_images.append(char2image[char])
        else:
            font = pygame.font.Font(os.path.join(Config.data.processed_path, "SourceHanSerif-Regular.ttc"),
                                    Config.model.char_image_size)
            try:
                rtext = font.render(char, True, (0, 0, 0), (255, 255, 255))
                img = pygame.surfarray.array3d(rtext)
                img = img.swapaxes(0, 1)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = img[1:, :]
            except Exception as e:
                print(e.args)
                img = np.ones((Config.model.char_image_size, Config.model.char_image_size)) * 255
            if img.shape[0] > img.shape[1]:
                pad_width = (img.shape[0] - img.shape[1]) // 2
                img = np.pad(img, [[0, 0], [pad_width, pad_width]], 'constant', constant_values=(255, 255))
            img = cv2.resize(img, (Config.model.char_image_size, Config.model.char_image_size))
            img = np.expand_dims(img, -1)
            char_images.append(img)
    return char_images


def word2id(words, vocab):
    word_id = [vocab.get(word, vocab['<UNK>']) for word in words]
    return word_id


def convert_to_example(inputs_images, label_id):
    inputs_images = np.array(inputs_images, np.uint8)

    data = {
        'inputs_images': _bytes_feature(inputs_images.tostring()),
        'label_id': _int64_feature(label_id)
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def preprocess(serialized):
    def parse_tfrecord(serialized):
        features = {
            'inputs_images': tf.FixedLenFeature([], tf.string),
            'label_id': tf.FixedLenFeature([Config.model.seq_length], tf.int64),
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        inputs_images = tf.decode_raw(parsed_example['inputs_images'], tf.uint8)
        inputs_images = tf.reshape(inputs_images,
                                   [Config.model.seq_length, Config.model.char_image_size, Config.model.char_image_size,
                                    1])
        label_id = parsed_example['label_id']
        return inputs_images, label_id

    inputs_images, label_id = parse_tfrecord(serialized)
    inputs_images = (inputs_images / 255 - 0.5) * 2
    return inputs_images, label_id


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
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            inputs_images = next_batch[0]
            label_id = next_batch[1]

            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: np.random.permutation(data)})

            return inputs_images, label_id

    return inputs, iterator_initializer_hook


def get_both_batch(dataA, dataB, buffer_size=1, batch_size=64, scope="train"):
    inputsA, iterator_initializer_hookA = get_dataset_batch(dataA, buffer_size, batch_size, scope)
    inputsB, iterator_initializer_hookB = get_dataset_batch(dataB, buffer_size, batch_size, scope)

    def inputs():
        for_inputs, for_labels = inputsA()
        back_inputs, back_labels = inputsB()

        return {'for_inputs': for_inputs, 'back_inputs': back_inputs}, \
               {'for_labels': for_labels, 'back_labels': back_labels}

    return inputs, [iterator_initializer_hookA, iterator_initializer_hookB]


def create_tfrecord():
    split_train_test()
    build_vocab()
    build_char_pkl()
    vocab = load_vocab()
    char2image = load_char2image()

    if not os.path.exists(os.path.join(Config.data.processed_path, 'for-tfrecord')):
        os.makedirs(os.path.join(Config.data.processed_path, 'for-tfrecord'))
    if not os.path.exists(os.path.join(Config.data.processed_path, 'back-tfrecord')):
        os.makedirs(os.path.join(Config.data.processed_path, 'back-tfrecord'))

    print('writing to tfrecord ...')
    for data in [Config.data.train_data, Config.data.test_data]:
        file = os.path.join(Config.data.processed_path, data)
        with open(file, encoding='utf8') as f:
            dataset_files = f.read().split()
            dataset_files = np.random.permutation(dataset_files)  # totally shuffled

        fidx = 0
        i = 0
        while i < len(dataset_files):

            if data is Config.data.train_data:
                for_tf_file = 'for-tfrecord/train_%d.tfrecord' % fidx
                back_tf_file = 'back-tfrecord/train_%d.tfrecord' % fidx
            else:
                for_tf_file = 'for-tfrecord/test.tfrecord'
                back_tf_file = 'back-tfrecord/test.tfrecord'
            for_tf_file = os.path.join(Config.data.processed_path, for_tf_file)
            back_tf_file = os.path.join(Config.data.processed_path, back_tf_file)

            with tf.python_io.TFRecordWriter(for_tf_file) as for_tf_writer, \
                    tf.python_io.TFRecordWriter(back_tf_file) as back_tf_writer:
                j = 0
                while i < len(dataset_files):
                    sys.stdout.write('\r>> converting %s %d/%d' % (data, i + 1, len(dataset_files)))
                    sys.stdout.flush()

                    for_samples, for_labels, back_samples, back_labels = read_file(dataset_files[i])
                    i += 1
                    for n in range(len(for_samples)):
                        inputs_images = word2image(for_samples[n], char2image)
                        label_id = word2id(for_labels[n], vocab)
                        example = convert_to_example(inputs_images, label_id)
                        serialized = example.SerializeToString()
                        for_tf_writer.write(serialized)

                        inputs_images = word2image(back_samples[n], char2image)
                        label_id = word2id(back_labels[n], vocab)
                        example = convert_to_example(inputs_images, label_id)
                        serialized = example.SerializeToString()
                        back_tf_writer.write(serialized)

                        j += 1
                    if j >= 5000 and data is Config.data.train_data:
                        break
                fidx += 1
            print('\n%s complete' % for_tf_file)
            print('%s complete' % back_tf_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/elmo.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    Config.data.dataset_path = os.path.expanduser(Config.data.dataset_path)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    create_tfrecord()
