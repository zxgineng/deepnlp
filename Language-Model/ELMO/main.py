import argparse
import tensorflow as tf
import os

import data_loader
from model import Model
from utils import Config


def run(mode, run_config):
    model = Model()
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=Config.train.model_dir,
        config=run_config)

    logginghook = tf.train.LoggingTensorHook({'for_loss': "Mean:0",
                                              'back_loss': "Mean_1:0",
                                              'step': 'global_step:0'}, every_n_iter=100)

    if mode == 'train':
        for_train_data = data_loader.get_tfrecord('for-tfrecord', 'train')
        for_val_data = data_loader.get_tfrecord('for-tfrecord', 'test')
        back_train_data = data_loader.get_tfrecord('back-tfrecord', 'train')
        back_val_data = data_loader.get_tfrecord('back-tfrecord', 'test')

        train_input_fn, train_input_hook = data_loader.get_both_batch(for_train_data, back_train_data, buffer_size=5000,
                                                                      batch_size=Config.train.batch_size, scope='val')
        val_input_fn, val_input_hook = data_loader.get_both_batch(for_val_data, back_val_data, batch_size=20,
                                                                  scope="val")

        while True:
            print('*' * 40)
            print("epoch", Config.train.epoch + 1, 'start')
            print('*' * 40)

            estimator.train(input_fn=train_input_fn, hooks=[logginghook] + train_input_hook)
            estimator.evaluate(input_fn=val_input_fn, hooks=[logginghook] + val_input_hook)

            Config.train.epoch += 1
            if Config.train.epoch == Config.train.max_epoch:
                break

    elif mode == 'eval':
        for_val_data = data_loader.get_tfrecord('for-tfrecord', 'test')
        back_val_data = data_loader.get_tfrecord('back-tfrecord', 'test')
        val_input_fn, val_input_hook = data_loader.get_both_batch(for_val_data, back_val_data, batch_size=20,
                                                                  scope="val")
        estimator.evaluate(input_fn=val_input_fn, hooks=[logginghook] + val_input_hook)


def main(mode):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
        model_dir=Config.train.model_dir,
        session_config=config,
        save_checkpoints_steps=Config.train.save_checkpoints_steps,
        log_step_count_steps=None)

    run(mode, run_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Mode (train)')
    parser.add_argument('--config', type=str, default='config/elmo.yml', help='config file name')

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    Config(args.config)
    Config.train.model_dir = os.path.expanduser(Config.train.model_dir)
    Config.data.processed_path = os.path.expanduser(Config.data.processed_path)

    print(Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(f" - {key}: {value}")

    main(args.mode)