import tensorflow as tf
import argparse
import os
import numpy as np

from utils import Config
from model import Model
import data_loader


def _make_estimator():
    params = tf.contrib.training.HParams(**Config.model.to_dict())
    run_config = tf.estimator.RunConfig(model_dir=Config.train.model_dir)

    model = Model()
    return tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=Config.train.model_dir,
        params=params,
        config=run_config)


def predict(ids, estimator):
    length = np.array([len(ids)])
    ids = np.expand_dims(np.array(ids), 0)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"id": ids, 'length': length},
        num_epochs=1,
        shuffle=False)

    result = next(estimator.predict(input_fn=predict_input_fn))
    return result['predictions']


def show(text, tokens):
    result = ''
    temp = ''
    for word, token in zip(text, tokens):
        if id2tag[token] == 'O':
            result += word
        elif id2tag[token][0] == 'S':
            result += ' {{' + word + '(:' + id2tag[token].split('-')[-1] + ')' + '}} '
        elif id2tag[token][0] == 'B':
            result += ' {{' + word
            temp = '(:' + id2tag[token].split('-')[-1] + ')'
        elif id2tag[token][0] == 'M':
            result += word
        elif id2tag[token][0] == 'E':
            result += word + temp + '}} '
        else:
            raise ValueError('invalid labels')
    print('result:')
    print(result)


def main():
    estimator = _make_estimator()
    vocab = data_loader.load_vocab("vocab.txt")
    while True:
        text = input('input text > ').strip()
        ids = data_loader.sentence2id(text, vocab)
        result = predict(ids, estimator)
        show(text, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/ner.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    id2tag = {i: k for i, k in enumerate(data_loader.load_tags())}

    main()
