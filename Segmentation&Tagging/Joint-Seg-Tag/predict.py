import tensorflow as tf
import os
import numpy as np

from utils import Config
from model import Model
import data_loader


class Predictor:
    def __init__(self):
        self.estimator = self._make_estimator()
        self.vocab = data_loader.load_vocab()
        self.tag = data_loader.load_tag()

    def _make_estimator(self):
        run_config = tf.estimator.RunConfig(model_dir=Config.train.model_dir)

        model = Model()
        return tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=Config.train.model_dir,
            config=run_config)

    def predict(self, text):
        if not isinstance(text, list):
            text = [text]
        length = np.array([len(t) for t in text])
        word_id = [data_loader.word2id(list(t), self.vocab) for t in text]
        word_id = tf.keras.preprocessing.sequence.pad_sequences(word_id, dtype='int64', padding='post')

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"word_id": word_id, 'length': length},
            batch_size=1024,
            num_epochs=1,
            shuffle=False)
        labels = list(self.estimator.predict(input_fn=predict_input_fn))
        return [data_loader.id2label(labels[i][:length[i]], self.tag) for i in range(len(text))]


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    Config('config/joint-seg-tag.yml')
    Config.train.model_dir = os.path.expanduser(Config.train.model_dir)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    p = Predictor()
    while True:
        text = input('input -> ')
        tokens = p.predict(text)[0]
        result = []
        word = text[0]
        tag = tokens[0][2:]
        for te, to in zip(text[1:], tokens[1:]):

            if to[0] == 'B' or to[0] == 'S':
                result.append((word, tag))
                word = ''
                tag = to[2:]
            word += te

        if len(word) != 0:
            result.append((word, tag))

        result = ' '.join(['/'.join(n) for n in result])
        print('result ->', result)
