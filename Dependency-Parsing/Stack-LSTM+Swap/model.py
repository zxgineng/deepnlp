import tensorflow as tf
import os
import time
import numpy as np

from utils import Config
from network import Graph
from data_loader import Token, Sentence, ArcStandardParser


class Model:
    def __init__(self, **kwargs):
        self.config = kwargs

    def train(self, input_fn):
        summary_writer = tf.contrib.summary.create_file_writer(self.config['model_dir'], flush_millis=10000)

        graph = Graph()
        global_step = tf.train.get_or_create_global_step()
        learning_rate = Config.train.initial_lr / (1 + Config.train.lr_decay * Config.train.epoch)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        ckpt_prefix = os.path.join(self.config['model_dir'], 'model.ckpt')
        checkpoint = tf.train.Checkpoint(model=graph, optimizer=optimizer, step_counter=global_step)
        checkpoint.restore(tf.train.latest_checkpoint(self.config['model_dir']))

        last_log_step = int(global_step)
        start_time = time.time()

        with summary_writer.as_default():
            while True:
                try:
                    inputs, targets = input_fn()
                except StopIteration:
                    break
                with tf.contrib.summary.record_summaries_every_n_global_steps(
                        self.config.get('save_summary_steps', 100),
                        global_step=global_step):
                    with tf.GradientTape() as tape:
                        logits = graph(inputs, mode='train')
                        reg_loss = Config.train.reg_scale * tf.reduce_sum(
                            [tf.nn.l2_loss(v) for v in graph.trainable_variables if 'embedding' not in v.name])
                        loss = tf.losses.sparse_softmax_cross_entropy(labels=targets['transition'], logits=logits)
                        loss = reg_loss + loss
                    grads = tape.gradient(loss, graph.trainable_variables)
                    clipped_gradients, _ = tf.clip_by_global_norm(grads, Config.train.max_gradient_norm)
                    optimizer.apply_gradients(zip(clipped_gradients, graph.trainable_variables),
                                              global_step=global_step)
                    tf.contrib.summary.scalar('loss', loss)

                if int((global_step - last_log_step)) % self.config.get('log_step_count_steps', 100) == 0:
                    end_time = time.time()
                    interval = end_time - start_time
                    start_time = end_time
                    tf.logging.info(
                        'loss = %.3f, step = %d (%.3f sec)' % (float(loss), global_step, float(interval)))

            checkpoint.save(ckpt_prefix)

    def evaluate(self, input_fn):
        summary_writer = tf.contrib.summary.create_file_writer(os.path.join(self.config['model_dir'], 'eval'))
        graph = Graph()
        global_step = tf.train.get_or_create_global_step()
        checkpoint = tf.train.Checkpoint(model=graph, step_counter=global_step)
        checkpoint.restore(tf.train.latest_checkpoint(self.config['model_dir']))
        parser = ArcStandardParser()

        correct_head_num = 0
        correct_dep_num = 0
        total_num = 0

        with summary_writer.as_default():
            while True:
                try:
                    inputs, targets = input_fn()
                except StopIteration:
                    break

                word = inputs['word'].numpy()
                pos = inputs['pos'].numpy()
                length = inputs['length'].numpy()
                head = targets['head'].numpy()
                dep_id = targets['dep_id'].numpy()

                batch_size = word.shape[0]
                total_sen = [Sentence([Token(n + 1, word[i][n].decode(), pos[i][n].decode(), None, None) for
                                       n in range(length[i])]) for i in range(batch_size)]

                while True:
                    batch_comp_word_id, batch_comp_pos_id, batch_comp_action_id, batch_comp_action_len, batch_buff_word_id, \
                    batch_buff_pos_id, batch_history_action_id, stack_length, buff_length, history_action_length = \
                        [], [], [], [], [], [], [], [], [], []
                    for sen in total_sen:
                        if not sen.terminate:
                            comp_word_id, comp_pos_id, comp_action_id, comp_action_len, buff_word_id, buff_pos_id, history_action_id \
                                = parser.extract_from_current_state(sen)
                            batch_comp_word_id.append(comp_word_id)
                            batch_comp_pos_id.append(comp_pos_id)
                            batch_comp_action_id.append(comp_action_id)
                            batch_comp_action_len.append(comp_action_len)
                            batch_buff_word_id.append(buff_word_id)
                            batch_buff_pos_id.append(buff_pos_id)
                            batch_history_action_id.append(history_action_id)
                            stack_length.append(len(comp_word_id))
                            buff_length.append(len(buff_word_id))
                            history_action_length.append(len(history_action_id))

                    batch_comp_word_id = tf.convert_to_tensor(
                        tf.keras.preprocessing.sequence.pad_sequences(batch_comp_word_id, dtype='int64',
                                                                      padding='post'))
                    batch_comp_pos_id = tf.convert_to_tensor(
                        tf.keras.preprocessing.sequence.pad_sequences(batch_comp_pos_id, dtype='int64',
                                                                      padding='post'))
                    batch_comp_action_id = tf.convert_to_tensor(
                        tf.keras.preprocessing.sequence.pad_sequences(batch_comp_action_id, dtype='int64',
                                                                      padding='post'))
                    batch_comp_action_len = tf.convert_to_tensor(
                        tf.keras.preprocessing.sequence.pad_sequences(batch_comp_action_len, dtype='int64',
                                                                      padding='post'))
                    batch_buff_word_id = tf.convert_to_tensor(
                        tf.keras.preprocessing.sequence.pad_sequences(batch_buff_word_id, dtype='int64',
                                                                      padding='post'))
                    batch_buff_pos_id = tf.convert_to_tensor(
                        tf.keras.preprocessing.sequence.pad_sequences(batch_buff_pos_id, dtype='int64',
                                                                      padding='post'))
                    batch_history_action_id = tf.convert_to_tensor(
                        tf.keras.preprocessing.sequence.pad_sequences(batch_history_action_id, dtype='int64',
                                                                      padding='post'))
                    inputs = {'buff_word_id': batch_buff_word_id, 'buff_pos_id': batch_buff_pos_id,
                              'history_action_id': batch_history_action_id,
                              'comp_word_id': batch_comp_word_id, 'comp_pos_id': batch_comp_pos_id,
                              'comp_action_id': batch_comp_action_id,
                              'comp_action_len': batch_comp_action_len, 'stack_length': stack_length,
                              'buff_length': buff_length,
                              'history_action_length': history_action_length}

                    logits = graph(inputs, mode='eval')
                    prob = tf.nn.softmax(logits)

                    logits_idx = 0
                    for sen in total_sen:
                        if not sen.terminate:
                            legal_transitions = parser.get_legal_transitions(sen)
                            transition = int(tf.argmax(np.array(legal_transitions) * prob[logits_idx], -1))
                            if transition not in [0, 1]:
                                parser.update_composition(sen, transition)  # update composition
                            parser.update_state_by_transition(sen, transition)  # update stack and buff
                            logits_idx += 1
                            if parser.terminal(sen):
                                sen.terminate = True
                    if [sen.terminate for sen in total_sen] == [True] * batch_size:
                        for i, sen in enumerate(total_sen):
                            gold_head = head[i, :length[i]]
                            pred_head = np.array([t.head_id for t in sen.tokens])
                            assert len(gold_head) == len(pred_head)
                            correct_head = gold_head == pred_head
                            correct_head_num += np.sum(correct_head)
                            gold_dep = dep_id[i, :length[i]]
                            pred_dep = np.array([t.dep for t in sen.tokens])
                            correct_dep = gold_dep == pred_dep
                            correct_dep_num += np.sum(np.logical_and(correct_head, correct_dep))
                            total_num += length[i]
                        break

            with tf.contrib.summary.always_record_summaries():
                uas = correct_head_num / total_num
                las = correct_dep_num / total_num
                tf.contrib.summary.scalar('UAS', uas)
                tf.contrib.summary.scalar('LAS', las)
                tf.logging.info(
                    'Evaluate result: step = %d, UAS = %.3f, LAS = %.3f' % (global_step, uas, las))
