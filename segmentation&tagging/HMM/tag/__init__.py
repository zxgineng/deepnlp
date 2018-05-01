import re
from .utils import pair
from .hmm_tag import hmm_cut

re_han_internal = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
re_skip_internal = re.compile("(\r\n|\s)")

re_eng = re.compile("[a-zA-Z0-9]+")
re_num = re.compile("[\.0-9]+")

re_eng1 = re.compile('^[a-zA-Z0-9]$', re.U)


class POSTokenizer(object):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.load_word_tag(open(self.tokenizer.dictionary, encoding='utf8'))

    def load_word_tag(self, f):
        """读取预存的词和词性"""
        self.word_tag_tab = {}
        f_name = f.name
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip()
                if not line:
                    continue
                word, _, tag = line.split(" ")
                self.word_tag_tab[word] = tag
            except Exception:
                raise ValueError(
                    'invalid POS dictionary entry in %s at Line %s: %s' % (f_name, lineno, line))
        f.close()

    def __cut_DAG_NO_HMM(self, sentence):
        """不使用HMM分词"""
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}
        self.tokenizer.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if re_eng1.match(l_word):
                buf += l_word
                x = y
            else:
                if buf:
                    yield pair(buf, 'eng')
                    buf = ''
                yield pair(l_word, self.word_tag_tab.get(l_word, 'x'))
                x = y
        if buf:
            yield pair(buf, 'eng')
            buf = ''

    def __cut_DAG(self, sentence):
        """根据DAG生成route,分割句子并标注词性"""
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}
        self.tokenizer.calc(sentence, DAG, route)
        x = 0
        # buf收集连续的单个字,把它们组合成字符串再交由 finalseg.cut函数来进行下一步分词
        buf = ''
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            # route中单字添加进buf
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    # 如一个单字后接一个词，直接返回单字pair
                    if len(buf) == 1:
                        yield pair(buf, self.word_tag_tab.get(buf, 'x'))
                    # 如不止一个单字 且未登录 利用HMM
                    elif not self.tokenizer.FREQ.get(buf):
                        recognized = hmm_cut(buf)
                        for t in recognized:
                            yield t
                    # 若已登录，则分开输出pair
                    else:
                        for elem in buf:
                            yield pair(elem, self.word_tag_tab.get(elem, 'x'))
                    buf = ''
                yield pair(l_word, self.word_tag_tab.get(l_word, 'x'))
            x = y
        # 句尾非词的情况
        if buf:
            if len(buf) == 1:
                yield pair(buf, self.word_tag_tab.get(buf, 'x'))
            elif not self.tokenizer.FREQ.get(buf):
                recognized = hmm_cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield pair(elem, self.word_tag_tab.get(elem, 'x'))

    def __cut_internal(self, sentence, HMM=True):

        blocks = re_han_internal.split(sentence)
        if HMM:
            cut_blk = self.__cut_DAG
        else:
            cut_blk = self.__cut_DAG_NO_HMM

        for blk in blocks:
            if re_han_internal.match(blk):
                for word in cut_blk(blk):
                    yield word
            # 非中文情况
            else:
                tmp = re_skip_internal.split(blk)
                for x in tmp:
                    if re_skip_internal.match(x):
                        yield pair(x, 'x')
                    else:
                        for xx in x:
                            if re_num.match(xx):
                                yield pair(xx, 'm')
                            elif re_eng.match(x):
                                yield pair(xx, 'eng')
                            else:
                                yield pair(xx, 'x')

    def cut(self, sentence, HMM=True):
        for w in self.__cut_internal(sentence, HMM=HMM):
            yield w


