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
        total_en1_end = []
        total_en2_end = []
        total_pos_1 = []
        total_pos_2 = []
        for n in inputs:
            sen = n[0]
            en1, en2 = n[1].split()
            en1_start = sen.index(en1)
            en1_end = en1_start + len(en1) - 1
            total_en1_end.append(en1_end)
            en2_start = sen.index(en2)
            en2_end = en2_start + len(en2) - 1
            total_en2_end.append(en2_end)
            word_id = data_loader.word2id(list(sen), self.vocab)
            total_sen.append(word_id)
            pos_1 = []
            pos_2 = []
            for n in range(len(sen)):
                if n < en1_start:
                    pos_1.append(n - en1_start)
                elif en1_start <= n <= en1_end:
                    pos_1.append(0)
                else:
                    pos_1.append(n - en1_end)

                if n < en2_start:
                    pos_2.append(n - en2_start)
                elif en2_start <= n <= en2_end:
                    pos_2.append(0)
                else:
                    pos_2.append(n - en2_end)
            total_pos_1.append(data_loader.pos_encode(pos_1))
            total_pos_2.append(data_loader.pos_encode(pos_2))

        total_sen = tf.keras.preprocessing.sequence.pad_sequences(total_sen,dtype='int64',padding='post')
        total_pos_1 = tf.keras.preprocessing.sequence.pad_sequences(total_pos_1, dtype='int64', padding='post')
        total_pos_2 = tf.keras.preprocessing.sequence.pad_sequences(total_pos_2, dtype='int64', padding='post')
        total_en1_end = np.array(total_en1_end)
        total_en2_end = np.array(total_en2_end)

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'word_id': total_sen, 'pos_1': total_pos_1, 'pos_2': total_pos_2,
                'en1_pos': total_en1_end, 'en2_pos': total_en2_end},
            batch_size=512,
            num_epochs=1,
            shuffle=False)
        results = list(self.estimator.predict(input_fn=predict_input_fn))
        results = data_loader.id2rel(results,self.rel_dict)
        return results


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    Config('config/pcnn-att.yml')
    Config.train.model_dir = os.path.expanduser(Config.train.model_dir)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    p = Predictor()
    while True:
        text = input('input text -> ')
        text = strQ2B(text)
        entity = input('input entity (separated by space) -> ')
        results = p.predict([[text,entity]])
        print('result ->', results[0])
