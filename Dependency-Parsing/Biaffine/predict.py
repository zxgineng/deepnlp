import tensorflow as tf
import os
import numpy as np

from utils import Config, strQ2B
from model import Model
from hooks import mst
import data_loader


class Predictor:
    def __init__(self):
        self.estimator = self._make_estimator()
        self.vocab = data_loader.load_vocab()
        self.pos_dict = data_loader.load_pos()
        self.dep_dict = data_loader.load_dep()

    def _make_estimator(self):
        run_config = tf.estimator.RunConfig(model_dir=Config.train.model_dir)

        model = Model()
        return tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=Config.train.model_dir,
            config=run_config)

    def predict(self, sen, pos):
        assert len(sen) == len(pos)
        if not isinstance(sen[0], list):
            sen = [sen]
        if not isinstance(pos[0], list):
            pos = [pos]

        length = np.array([len(s) + 1 for s in sen])
        word_id = [[0] + data_loader.word2id(s, self.vocab) for s in sen]
        pos_id = [[0] + data_loader.pos2id(p, self.pos_dict) for p in pos]
        word_id = tf.keras.preprocessing.sequence.pad_sequences(word_id, dtype='int64', padding='post')
        pos_id = tf.keras.preprocessing.sequence.pad_sequences(pos_id, dtype='int64', padding='post')

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"word_id": word_id, 'pos_id': pos_id, 'length': length},
            batch_size=128,
            num_epochs=1,
            shuffle=False)
        pred = list(self.estimator.predict(input_fn=predict_input_fn))
        results = []
        for i in range(len(pred)):
            result = []
            arc = mst(pred[i]['arc_logits'][:length[i], :length[i]])[1:]
            label = np.argmax(pred[i]['label_logits'][range(1, length[i]), arc, :], -1)
            label = data_loader.id2dep(label, self.dep_dict)
            [result.append((w, p, str(a), l)) for w, p, a, l in zip(sen[i], pos[i], arc, label)]
            results.append(result)
        return results


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    Config('config/biaffine.yml')
    Config.train.model_dir = os.path.expanduser(Config.train.model_dir)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    p = Predictor()
    # words = '令 人 遗憾 的 是 , 他 至今 未 能 找到 令 自己 满意 的 答案 。'
    # pos = 'VV NN VV DEC VC PU PN AD AD VV VV VV PN VV DEC NN PU'
    while True:
        text = input('input words (separated by space) -> ')
        text = strQ2B(text)
        words = text.split(' ')
        pos = input('input tags (separated by space) -> ')
        pos = strQ2B(pos)
        pos = pos.split(' ')
        results = p.predict(words, pos)
        print('result ->')
        print('\n'.join(['\t'.join(n) for n in results[0]]) + '\n')
