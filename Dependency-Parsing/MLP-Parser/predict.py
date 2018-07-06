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


def predict(word, pos, dep, estimator):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"word": word, 'pos': pos, 'dep': dep},
        num_epochs=1,
        batch_size=word.shape[0],
        shuffle=False)

    result = next(estimator.predict(input_fn=predict_input_fn, yield_single_examples=False))
    return result['predictions']


def show(sentences, dep_vocab):
    id2dep = {v: k for k, v in dep_vocab.items()}
    for sentence in sentences:
        print('*' * 20)
        print('text: ', ' '.join([token.word for token in sentence.tokens]))
        for dep in sentence.predicted_dependencies:
            print(dep[0].word, '--->', dep[1].word, '\t', id2dep[(dep[2] - 1) // 2].split(':')[1])


def compute_dependencies(sentences, word_vocab, pos_vocab, dep_vocab):
    estimator = _make_estimator()

    enable_features = [0 if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1 for sentence in
                       sentences]
    enable_count = np.count_nonzero(enable_features)

    while enable_count > 0:
        curr_sentences = [sentence for i, sentence in enumerate(sentences) if enable_features[i] == 1]

        curr_inputs = [
            data_loader.extract_for_current_state(sentence, word_vocab, pos_vocab,
                                                  dep_vocab) for sentence in curr_sentences]
        # create batch
        word_inputs = np.array([curr_inputs[i][0] for i in range(len(curr_inputs))])
        pos_inputs = np.array([curr_inputs[i][1] for i in range(len(curr_inputs))])
        dep_inputs = np.array([curr_inputs[i][2] for i in range(len(curr_inputs))])

        predictions = predict(word_inputs, pos_inputs, dep_inputs, estimator)
        legal_labels = np.array([sentence.get_legal_labels() for sentence in curr_sentences],
                                dtype=np.float32)
        legal_transitions = np.argmax(predictions + 1000 * legal_labels, axis=-1)
        # update left/right children
        [sentence.update_child_dependencies(transition) for (sentence, transition) in
         zip(curr_sentences, legal_transitions) if transition != 0]
        # update state
        [sentence.update_state_by_transition(legal_transition, gt=False) for (sentence, legal_transition) in
         zip(curr_sentences, legal_transitions)]

        enable_features = [0 if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1 for sentence in
                           sentences]
        enable_count = np.count_nonzero(enable_features)

    return sentences


def main(file):
    data = data_loader.read_file(file, False)
    word_vocab = data_loader.load_pkl('word2idx.pkl')
    pos_vocab = data_loader.load_pkl('pos2idx.pkl')
    dep_vocab = data_loader.load_pkl('dep2idx.pkl')
    sentences = compute_dependencies(data, word_vocab, pos_vocab, dep_vocab)
    show(sentences, dep_vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/mlp-parser.yml',
                        help='config file name')
    parser.add_argument('--file', type=str, default='test.conll', help='file name, must follow CoNLL data format')
    args = parser.parse_args()

    Config(args.config)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    main(args.file)
