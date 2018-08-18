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
        self.rel_dict = data_loader.load_rel()

    def _make_estimator(self):
        run_config = tf.estimator.RunConfig(model_dir=Config.train.model_dir)

        model = Model()
        return tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=Config.train.model_dir,
            config=run_config)

    def predict(self, inputs):
        total_sen = []
        total_indicator = []
        length = []
        for n in inputs:
            sen = n[0]
            en1, en2 = n[1].split()
            word_id = data_loader.word2id(list(sen), self.vocab)
            total_sen.append(word_id)
            en_indicator = [0] * len(sen)
            en_indicator[sen.index(en1):sen.index(en1) + len(en1)] = [1] * len(en1)
            en_indicator[sen.index(en2):sen.index(en2) + len(en2)] = [-1] * len(en2)
            total_indicator.append(en_indicator)
            length.append(len(sen))

        total_sen = tf.keras.preprocessing.sequence.pad_sequences(total_sen,dtype='int64',padding='post')
        total_indicator = tf.keras.preprocessing.sequence.pad_sequences(total_indicator, dtype='int64', padding='post')
        length = np.array(length)

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'word_id': total_sen, 'en_indicator': total_indicator, 'length': length},
            batch_size=512,
            num_epochs=1,
            shuffle=False)
        results = list(self.estimator.predict(input_fn=predict_input_fn))
        results = data_loader.id2rel(results,self.rel_dict)
        return results


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    Config('config/bigru-adv-soft_label.yml')
    Config.train.model_dir = os.path.expanduser(Config.train.model_dir)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    p = Predictor()
    while True:
        text = input('input text -> ')
        text = strQ2B(text)
        entity = input('input entity (separated by space) -> ')
        results = p.predict([[text,entity]])
        print('result ->', results[0])
