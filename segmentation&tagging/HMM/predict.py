from seg import dt
from tag import POSTokenizer
import argparse

seg = dt.cut

tag_dt = POSTokenizer(dt)

tag = tag_dt.cut



def predict(args):
    sentence = args.sentence
    if args.mode == 'seg':
        result = list(seg(sentence))
    elif args.mode == 'tag':
        result = list(tag(sentence))
    else:
        raise ValueError('invalid mode')
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default='seg',
                        choices=['seg', 'tag'],
                        help='Mode (seg/tag)')
    parser.add_argument('sentence', type=str, help='input sentence')
    args = parser.parse_args()
    predict(args)
