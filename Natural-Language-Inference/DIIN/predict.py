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
        self.pos_dict = data_loader.load_pos()

    def _make_estimator(self):
        run_config = tf.estimator.RunConfig(model_dir=Config.train.model_dir)

        model = Model()
        return tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=Config.train.model_dir,
            config=run_config)

    def predict(self, inputs):
        premise_words = [data_loader.pad_to_fixed_len(n[0]) for n in inputs]
        premise_tags = [data_loader.pad_to_fixed_len(n[1]) for n in inputs]
        hypothesis_words = [data_loader.pad_to_fixed_len(n[2]) for n in inputs]
        hypothesis_tags = [data_loader.pad_to_fixed_len(n[3]) for n in inputs]

        p_word_id = np.array([data_loader.word2id(s, self.vocab) for s in premise_words], np.int64)
        p_pos_id = np.array([data_loader.pos2id(p, self.pos_dict) for p in premise_tags], np.int64)
        h_word_id = np.array([data_loader.word2id(s, self.vocab) for s in hypothesis_words], np.int64)
        h_pos_id = np.array([data_loader.pos2id(p, self.pos_dict) for p in hypothesis_tags], np.int64)
        p_char_images = np.array([data_loader.word2image(s) for s in premise_words], np.float32) / 127.5 - 1
        h_char_images = np.array([data_loader.word2image(s) for s in hypothesis_words], np.float32) / 127.5 - 1

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'p_word_id': p_word_id, 'p_pos_id': p_pos_id, 'h_word_id': h_word_id, 'h_pos_id': h_pos_id,
               'p_char_images': p_char_images, 'h_char_images': h_char_images},
            batch_size=20,
            num_epochs=1,
            shuffle=False)
        results = list(self.estimator.predict(input_fn=predict_input_fn))
        return results


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    Config('config/diin.yml')
    Config.train.model_dir = os.path.expanduser(Config.train.model_dir)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    p = Predictor()
    word_map = {1: '相同意图', 0: '不同意图'}
    while True:
        origin_premise = input('input premise words (separated by space) -> ')
        premise_tags = input('input premise tags (separated by space) -> ')
        premise_words = strQ2B(origin_premise).split(' ')
        premise_tags = premise_tags.split(' ')

        origin_hypothesis = input('input hypothesis words (separated by space) -> ')
        hypothesis_tags = input('input hypothesis tags (separated by space) -> ')
        hypothesis_words = strQ2B(origin_hypothesis).split(' ')
        hypothesis_tags = hypothesis_tags.split(' ')

        results = p.predict([[premise_words, premise_tags, hypothesis_words, hypothesis_tags]])
        print('result ->')
        print('句子一:', ''.join(premise_words))
        print('句子二:', ''.join(hypothesis_words))
        print('结果:', word_map[results[0]])
