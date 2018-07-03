import tensorflow as tf
import os

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
        word_id = [data_loader.word2id(list(t), self.vocab) for t in text]
        word_id = tf.keras.preprocessing.sequence.pad_sequences(word_id, maxlen=Config.data.max_sequence_length,dtype='int64', padding='post',truncating='post')

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=word_id,
            batch_size=512,
            num_epochs=1,
            shuffle=False)
        labels = list(self.estimator.predict(input_fn=predict_input_fn))
        return [data_loader.id2label(labels[i], self.tag) for i in range(len(text))]


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    Config('config/dpcnn.yml')
    Config.train.model_dir = os.path.expanduser(Config.train.model_dir)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    p = Predictor()
    print('总分类包括: 体育, 娱乐, 家居, 彩票, 房产, 教育, 时尚, 时政, 星座, 游戏, 社会, 科技, 股票, 财经')
    while True:
        text = input('input -> ')
        result = p.predict(text)[0]
        print('result ->', result)
