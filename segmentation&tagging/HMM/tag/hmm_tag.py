import re

from .utils import pair
from .char_state_tab import P as char_state_tab_P
from .prob_start import P as start_P
from .prob_trans import P as trans_P
from .prob_emit import P as emit_P

MIN_FLOAT = -3.14e100
MIN_INF = float("-inf")
re_han_detail = re.compile("([\u4E00-\u9FD5]+)")
re_skip_detail = re.compile("([\.0-9]+|[a-zA-Z0-9]+)")
re_eng = re.compile("[a-zA-Z0-9]+")
re_num = re.compile("[\.0-9]+")

def __cut(sentence):
    """使用viterbi分词"""
    prob, pos_list = viterbi(
        sentence, char_state_tab_P, start_P, trans_P, emit_P)
    begin, nexti = 0, 0
    # 根据结果生成pair
    for i, char in enumerate(sentence):
        pos = pos_list[i][0]
        if pos == 'B':
            begin = i
        elif pos == 'E':
            yield pair(sentence[begin:i + 1], pos_list[i][1])
            nexti = i + 1
        elif pos == 'S':
            yield pair(char, pos_list[i][1])
            nexti = i + 1
    if nexti < len(sentence):
        yield pair(sentence[nexti:], pos_list[nexti][1])


def hmm_cut(sentence):
    """分离汉字与非汉字，对汉字使用HMM"""
    blocks = re_han_detail.split(sentence)
    for blk in blocks:
        # 汉字使用HMM
        if re_han_detail.match(blk):
            for word in __cut(blk):
                yield word
        # 非汉字按各自情况输出词性
        else:
            tmp = re_skip_detail.split(blk)
            for x in tmp:
                if x:
                    if re_num.match(x):
                        yield pair(x, 'm')
                    elif re_eng.match(x):
                        yield pair(x, 'eng')
                    else:
                        yield pair(x, 'x')


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    mem_path = [{}]
    all_states = trans_p.keys()
    # 获取初始值,只获取obs[0]对应的可能的状态
    for y in states.get(obs[0], all_states):
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        mem_path[0][y] = ''

    for t in range(1, len(obs)):
        V.append({})
        mem_path.append({})
        # 在t-1的状态中，提取出能转移出下一状态的状态
        prev_states = [x for x in mem_path[t - 1].keys() if len(trans_p[x]) > 0]
        # t-1的状态中对应的所有可能的下一状态
        prev_states_expect_next = set(
            (y for x in prev_states for y in trans_p[x].keys()))
        obs_states = set(
            states.get(obs[t], all_states)) & prev_states_expect_next

        if not obs_states:
            obs_states = prev_states_expect_next if prev_states_expect_next else all_states

        for y in obs_states:
            prob, state = max((V[t - 1][y0] + trans_p[y0].get(y, MIN_INF) +
                               emit_p[y].get(obs[t], MIN_FLOAT), y0) for y0 in prev_states)
            V[t][y] = prob
            mem_path[t][y] = state

    last = [(V[-1][y], y) for y in mem_path[-1].keys()]
    prob, state = max(last)

    route = [None] * len(obs)
    i = len(obs) - 1
    while i >= 0:
        route[i] = state
        state = mem_path[i][state]
        i -= 1
    return (prob, route)