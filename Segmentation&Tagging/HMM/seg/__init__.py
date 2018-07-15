import re
from math import log
from .hmm_seg import hmm_cut

DEFAULT_DICT = "dict.txt"

re_eng = re.compile('[a-zA-Z0-9]', re.U)

re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)", re.U)
re_skip_default = re.compile("(\r\n|\s)", re.U)

class Tokenizer():

    def __init__(self):
        self.dictionary = DEFAULT_DICT
        self.FREQ = {}
        self.total = 0
        self.initialized = False
        # self.user_word_tag_tab = {}

    def __cut_DAG_NO_HMM(self, sentence):
        """不使用HMM,未登录词直接单个输出"""
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            # 英语按原顺序合并输出
            if re_eng.match(l_word) and len(l_word) == 1:
                buf += l_word
                x = y
            else:
                if buf:
                    yield buf
                    buf = ''
                yield l_word
                x = y
        if buf:
            yield buf
            buf = ''

    def __cut_DAG(self, sentence):
        """根据DAG生成roune,分割句子"""
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
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
                    # 如一个单字后接一个词，直接返回单字
                    if len(buf) == 1:
                        yield buf
                        buf = ''
                    # 如不止一个单字
                    else:
                        # 未登录词,利用HMM
                        if not self.FREQ.get(buf):
                            recognized = hmm_cut(buf)
                            for t in recognized:
                                yield t
                        # 若已登录，则分开输出
                        else:
                            for elem in buf:
                                yield elem
                        buf = ''
                yield l_word
            x = y
        # 句尾非词的情况
        if buf:
            if len(buf) == 1:
                yield buf
            elif not self.FREQ.get(buf):
                recognized = hmm_seg.cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

    def calc(self, sentence, DAG, route):
        N = len(sentence)
        route[N] = (0, 0)
        # 对概率值取对数
        logtotal = log(self.total)
        # 从后往前遍历句子 反向计算最大概率
        for idx in range(N - 1, -1, -1):
            # route[idx] = max([ (概率对数，词语末字位置) for x in DAG[idx] ])
            # 以idx:(概率对数最大值，词语末字位置)键值对形式保存在route中
            # [x+1][0]即表示取句子x+1位置对应元组(概率对数，词语末字位置)的概率对数
            # 取sentence[idx:x + 1]与其之后一个词的联合概率最大值
            route[idx] = max((log(self.FREQ.get(sentence[idx:x + 1]) or 1) -
                              logtotal + route[x + 1][0], x) for x in DAG[idx])

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def cut(self, sentence, HMM=True):

        re_han = re_han_default
        re_skip = re_skip_default

        if HMM:
            cut_block = self.__cut_DAG
        else:
            cut_block = self.__cut_DAG_NO_HMM
        # 分离有效文字
        blocks = re_han.split(sentence)
        for blk in blocks:
            if not blk:
                continue
            # 判断是否有文字
            if re_han.match(blk):
                # 对句子进行分割
                for word in cut_block(blk):
                    yield word
            # 如只有空格回车
            else:
                tmp = re_skip.split(blk)
                for x in tmp:
                    if re_skip.match(x):
                        yield x
                    else:
                        for xx in x:
                            yield xx

    # DAG中是以{key:list,...}的字典结构存储
    # key是字的开始位置
    def gen_pfdict(self, f):
        """读取每个词的次数和总次数"""
        lfreq = {}
        ltotal = 0
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            word, freq = line.split(' ')[:2]
            freq = int(freq)
            lfreq[word] = freq
            ltotal += freq
            for ch in range(len(word)):
                wfrag = word[:ch + 1]
                if wfrag not in lfreq:
                    lfreq[wfrag] = 0
        f.close()
        return lfreq, ltotal

    def get_DAG(self, sentence):
        """生成DAG"""
        self.check_initialized()
        DAG = {}
        N = len(sentence)
        for k in range(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N and frag in self.FREQ:
                if self.FREQ[frag]:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

    def initialize(self):
        self.FREQ, self.total = self.gen_pfdict(open(self.dictionary,encoding='utf8'))
        self.initialized = True

dt = Tokenizer()

