import argparse
import tensorflow as tf
import os

import data_loader
from model import Model
from utils import Config


def run(run_config):
    model = Model()
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=Config.train.model_dir,
        config=run_config)

    train_data = data_loader.get_tfrecord('train')
    train_input_fn, train_input_hook = data_loader.get_dataset_batch(train_data, buffer_size=5000,
                                                                     batch_size=Config.train.batch_size,
                                                                     scope="train")

    estimator.train(input_fn=train_input_fn, max_steps=Config.train.max_steps, hooks=[train_input_hook])


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
        model_dir=Config.train.model_dir,
        session_config=config,
        save_checkpoints_steps=Config.train.save_checkpoints_steps)

    run(run_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/bigru-adv-soft_label.yml', help='config file name')

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

    main()
