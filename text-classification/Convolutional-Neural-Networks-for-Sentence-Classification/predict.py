import argparse
import os
import sys

from utils import Config
import numpy as np
import tensorflow as tf

import data_loader
from model import Model


cls = {'体育':0,
       '娱乐':1,
       '家具':2,
       '房产':3,
       '教育':4,
       '时尚':5,
       '时政':6,
       '游戏':7,
       '科技':8,
       '财经':9}

def predict(ids):

    X = np.array(ids, dtype=np.int32)
    X = np.expand_dims(X,0)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"inputs": X},
            num_epochs=1,
            shuffle=False)

    estimator = _make_estimator()
    result = estimator.predict(input_fn=predict_input_fn)

    prediction = next(result)["prediction"]
    return prediction


def _make_estimator():
    params = tf.contrib.training.HParams(**Config.model.to_dict())
    run_config = tf.contrib.learn.RunConfig(model_dir=Config.train.model_dir)

    model = Model()
    return tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=Config.train.model_dir,
            params=params,
            config=run_config)


def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


def main():
    vocab = data_loader.load_vocab("vocab.txt")
    idx2cls = dict(zip(cls.values(),cls.keys()))
    print("Typing news \n")

    while True:
        sentence = _get_user_input()
        ids = data_loader.sentence2id(vocab, sentence)
        result = predict(ids)
        print(idx2cls[result])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/thunews.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    main()
