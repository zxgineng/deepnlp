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
    tokens = [id2tag[token] for token in tokens]
    result = ''
    temp = ''
    for word, token in zip(text, tokens):
        if token.split('-')[-1] == 'w':
            result += word
            continue
        if token[0] == 'S':
            result += word + '/' + token.split('-')[-1] + ' '
        elif token[0] == 'B':
            result += word
            temp = token.split('-')[-1]
        elif token[0] == 'M':
            result += word
        elif token[0] == 'E':
            if temp:
                result += word + '/' + temp + ' '
            else:
                result += word
        else:
            raise ValueError('invalid labels')

    print('result > ',result)


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
    parser.add_argument('--config', type=str, default='config/people1998.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    id2tag = {i: k for i, k in enumerate(data_loader.load_tags('tags.txt'))}

    main()