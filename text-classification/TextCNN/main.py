import argparse
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import math

import data_loader
from model import Model
from utils import Config


def run(mode, run_config, params):
    model = Model()
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=Config.train.model_dir,
        params=params,
        config=run_config)

    if Config.train.debug:
        debug_hooks = tf_debug.LocalCLIDebugHook()
        hooks = [debug_hooks]
    else:
        hooks = []

    loss_hooks = tf.train.LoggingTensorHook({'loss': 'loss/loss/value:0', 'step': 'global_step:0'},
                                            every_n_iter=Config.train.check_hook_n_iter)

    if mode == 'train':
        train_data = data_loader.make_data_set(mode)
        train_input_fn, train_input_hook = data_loader.get_dataset_batch(train_data, batch_size=Config.model.batch_size,
                                                                         scope="train")
        hooks.extend([train_input_hook, loss_hooks])
        estimator.train(input_fn=train_input_fn, hooks=hooks, max_steps=Config.train.max_steps)

    elif mode == 'train_and_val':
        train_data, val_data = data_loader.make_data_set(mode)
        train_input_fn, train_input_hook = data_loader.get_dataset_batch(train_data, batch_size=Config.model.batch_size,
                                                                         scope="train")
        val_input_fn, val_input_hook = data_loader.get_dataset_batch(val_data, batch_size=Config.model.batch_size,
                                                                     scope="validation")
        hooks.extend([train_input_hook, loss_hooks])
        for n in range(math.ceil(Config.train.max_steps / Config.train.min_eval_frequency)):
            estimator.train(input_fn=train_input_fn, hooks=hooks,
                            steps=min(Config.train.min_eval_frequency,Config.train.max_steps - n * Config.train.min_eval_frequency))
            estimator.evaluate(input_fn=val_input_fn, hooks=[val_input_hook])

    elif mode == 'test':
        test_data = data_loader.make_data_set(mode)
        test_input_fn, test_input_hook = data_loader.get_dataset_batch(test_data, batch_size=Config.model.batch_size,
                                                                       scope="test")
        hooks.extend([test_input_hook])
        estimator.evaluate(input_fn=test_input_fn, hooks=hooks)

    else:
        raise ValueError('no %s mode' % (mode))


def main(mode):
    params = tf.contrib.training.HParams(**Config.train.to_dict())

    run_config = tf.estimator.RunConfig(
        model_dir=Config.train.model_dir,
        save_checkpoints_steps=Config.train.save_checkpoints_steps)

    run(mode, run_config, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'train_and_val'],
                        help='Mode (train/test/train_and_val)')
    parser.add_argument('--config', type=str, default='config/thunews.yml', help='config file name')

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    Config(args.config)
    print(Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(f" - {key}: {value}")

    main(args.mode)
