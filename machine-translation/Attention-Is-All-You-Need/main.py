import argparse
import tensorflow as tf
from tensorflow.python import debug as tfdbg

import data_loader
from model import Model
import utils
from utils import Config


def experiment_fn(run_config, params):
    model = Model()
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=Config.train.model_dir,
        params=params,
        config=run_config)
    # get the dictionary of tokens and their corresponding index
    source_vocab = data_loader.load_vocab("source_vocab")
    target_vocab = data_loader.load_vocab("target_vocab")
    # covert dictionary of tokens and their corresponding index into dictionary of index and tokens
    # save into Config
    Config.data.rev_source_vocab = utils.get_rev_vocab(source_vocab)
    Config.data.rev_target_vocab = utils.get_rev_vocab(target_vocab)
    Config.data.source_vocab_size = len(source_vocab)
    Config.data.target_vocab_size = len(target_vocab)
    # load all seq and pad to max seq length
    # train_data:(train_enc, train_dec)
    train_data, test_data = data_loader.make_train_and_test_set()
    # get input func and initial hook
    train_input_fn, train_input_hook = data_loader.get_dataset_batch(train_data,
                                                                     batch_size=Config.model.batch_size,
                                                                     scope="train")
    test_input_fn, test_input_hook = data_loader.get_dataset_batch(test_data,
                                                                   batch_size=Config.model.batch_size,
                                                                   scope="test")

    train_hooks = [train_input_hook]
    help_hooks = tf.train.LoggingTensorHook({'loss': 'loss/sequence-loss/truediv:0', 'step': 'global_step:0'},
                                            every_n_iter=Config.train.check_hook_n_iter)
    train_hooks = train_hooks + [help_hooks]

    if Config.train.debug:
        train_hooks.append(tfdbg.LocalCLIDebugHook())

    eval_hooks = [test_input_hook]
    if Config.train.debug:
        eval_hooks.append(tfdbg.LocalCLIDebugHook())

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=test_input_fn,
        train_steps=Config.train.train_steps,
        min_eval_frequency=Config.train.min_eval_frequency,
        train_monitors=train_hooks,
        eval_hooks=eval_hooks
    )
    return experiment


def main(mode):

    params = tf.contrib.training.HParams(**Config.train.to_dict())

    run_config = tf.contrib.learn.RunConfig(
        model_dir=Config.train.model_dir,
        save_checkpoints_steps=Config.train.save_checkpoints_steps)

    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=mode,
        hparams=params
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default='train', help='Mode (train/train_and_evaluate)')
    parser.add_argument('--config', type=str, default='config/iwslt15-en-vi.yml', help='config file name')

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)
    Config(args.config)
    print(Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(f" - {key}: {value}")


    main(args.mode)
