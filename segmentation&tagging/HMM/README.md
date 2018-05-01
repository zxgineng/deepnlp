# 基于隐马尔可夫模型HMM的中文分词和词性标注



## Requirements

- Python 3


## Project Structure


    ├── seg                     # 分词功能
        ├── __init__.py             # 分词功能主逻辑
        ├── hmm_seg.py              # 使用HMM分词
        ├── prob_emit.py            # 观察概率
        ├── prob_start.py           # 初始概率
        └── prob_trans.py           # 状态概率
    ├── tag                     # 词性标注功能
        ├── __init__.py             # 词性标注功能主逻辑
        ├── hmm_seg.py              # 使用HMM词性标注
        ├── prob_emit.py            # 观察概率
        ├── prob_start.py           # 初始概率
        ├── prob_trans.py           # 状态概率
        └── utils.py           # 辅助工具
    ├── predict.py              # 运行程序
    └── dict.txt                # 根据人民日报等公开语料进行统计后的词库
    


## Run


```
python predict.py --mode seg 需要分词的句子
```
```
python predict.py --mode tag 需要词性标注的句子
```


## Example

```
python predict.py --mode seg 这是一个中文分词的例子
['这是', '一个', '中文', '分词', '的', '例子']
```
```
python predict.py --mode seg 使用HMM模型对未登 录词进行分词
['使用', 'HMM', '模型', '对', '未', '登录', '词', '进行', '分词']
```
```
python predict.py --mode tag 字典加上HMM或CRF是传统分词模型
[pair('字典', 'n'), pair('加上', 'v'), pair('HMM', 'eng'), pair('或', 'c'), pair('CRF', 'eng'), pair('是', 'v'), pair('传统', 'n'), pair('分词', 'n'), pair('模型', 'n')]
```
```
python predict.py --mode tag 深度学习方面可以使用LSTM+CRF模型
[pair('深度', 'ns'), pair('学习', 'v'), pair('方面', 'n'), pair('可以', 'c'), pair('使用', 'v'), pair('LSTM', 'eng'), pair('+', 'x'), pair('CRF', 'eng'), pair('模型', 'n')]

```

