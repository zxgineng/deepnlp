import tensorflow as tf

from utils import Config


class SwitchOptimizerHook(tf.train.SessionRunHook):
    def __init__(self):
        self.min_loss = Config.train.get('min_loss', 1000)
        self.epoch = Config.train.get('min_loss_epoch', 0)

    def after_run(self, run_context, run_values):
        loss = run_values.results
        if loss < self.min_loss:
            Config.train.min_loss = loss
            Config.train.min_loss_epoch = Config.train.epoch

    def end(self, session):
        if self.epoch and (Config.train.epoch - self.epoch) >= 3:
            Config.train.switch_optimizer = 1