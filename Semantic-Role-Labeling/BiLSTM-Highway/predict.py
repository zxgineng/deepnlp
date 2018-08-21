import tensorflow as tf
import os
import numpy as np

from utils import Config, strQ2B
from model import Model
import data_loader


class Predictor:
    def __init__(self):
        self.estimator = self._make_estimator()
        self.vocab = data_loader.load_vocab()
        self.tag_dict = data_loader.load_tag()
        self.label_dict = data_loader.load_label()

    def _make_estimator(self):
        run_config = tf.estimator.RunConfig(model_dir=Config.train.model_dir)

        model = Model()
        return tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=Config.train.model_dir,
            config=run_config)

    def predict(self, inputs):
        word_id = []
        tag_id = []
        predicate = []
        length = []
        for n in inputs:
            word_id.append(data_loader.word2id(n[0], self.vocab))
            tag_id.append(data_loader.tag2id(n[1], self.tag_dict))
            length.append(len(n[0]))
            temp = [0] * len(n[0])
            temp[n[0].index(n[2])] = 1
            predicate.append(temp)

        length = np.array(length)
        word_id = tf.keras.preprocessing.sequence.pad_sequences(word_id, dtype='int64', padding='post')
        tag_id = tf.keras.preprocessing.sequence.pad_sequences(tag_id, dtype='int64', padding='post')
        predicate = tf.keras.preprocessing.sequence.pad_sequences(predicate, dtype='int64', padding='post')

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'word_id': word_id, 'tag_id': tag_id, 'predicate': predicate, 'length': length},
            batch_size=512,
            num_epochs=1,
            shuffle=False)
        labels = list(self.estimator.predict(input_fn=predict_input_fn))
        return [data_loader.id2label(labels[i][:length[i]], self.label_dict) for i in range(len(inputs))]


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    Config('config/bilstm-highway.yml')
    Config.train.model_dir = os.path.expanduser(Config.train.model_dir)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    p = Predictor()
    while True:
        text = input('input words (separated by space) -> ')
        text = strQ2B(text)
        words = text.split(' ')
        tags = input('input tags (separated by space) -> ')
        tags = strQ2B(tags)
        tags = tags.split(' ')
        predicate = input('input predicate -> ')
        results = p.predict([[words, tags, predicate]])
        print('result ->')
        for i in range(len(words)):
            print("{:\u3000<5} {:<5}".format(words[i], results[0][i]))
