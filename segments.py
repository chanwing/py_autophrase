from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import generator_stop

import os
import operator
import functools
import itertools
import collections
import logging
import tqdm
import bounter
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from pyhanlp import HanLP
from numpy import ma

from utils import ngram
from utils import preprocess
from utils import split_iter

logging.basicConfig(format='%(asctime)s : %(message)s',
                    level=logging.DEBUG)


class AutoPhrase:
  FNAME_POSTAG = os.path.normpath(os.path.join(os.path.dirname(__file__), 'tags.txt'))

  def __init__(self, omega, quality, length_threshold):
    """AutoPhrase

    :param omega::Omega
    :param quality::Quality
    """
    if not isinstance(omega, Omega):
      err = 'argument "omega" must be Omega, ' \
            'got {}'.format(type(omega))
      raise TypeError(err)

    if not isinstance(quality, Quality):
      err = 'argument "quality" must be Quality, ' \
            'got {}'.format(type(quality))
      raise TypeError(err)

    self.length_threshold = length_threshold

    # 语料库
    self.C = omega
    # 短语生成模型
    self.Q = quality

    self._delta = None
    self._theta = None
    self._param_delta = None
    self._param_theta = None
    self._phrases = None

    self._trace_param_delta = list()
    self._trace_param_theta = list()

  @property
  def phrases(self):
    if self._phrases is not None:
      return self._phrases

  @property
  def delta(self):
    if self._delta is not None:
      return self._delta

  @property
  def theta(self):
    if self._theta is not None:
      return self._theta

  @property
  def param_delta(self):
    if self._param_delta is not None:
      return self._param_delta

  @property
  def param_theta(self):
    if self._param_theta is not None:
      return self._param_theta

  def train(self, memory_size_mb):
    """
    基于维特比算法的训练函数

    Input: Corpus Ω and phrase quality Q.
    Output: θu and δ(tx, ty).
    """
    logging.debug('Viterbi Training.')

    # initialize θ with normalized raw frequencies in the corpus
    self.init_parameters(memory_size_mb=memory_size_mb)

    while decreasing(self._trace_param_theta):
      while decreasing(self._trace_param_delta):
        current_B = self.PGPS()
        self.update_delta(current_B, memory_size_mb)

      current_B = self.PGPS()
      self.update_theta(current_B, memory_size_mb)

    logging.debug('both theta and delta not converge.')

    # 获取纠正质量得分后的短语
    phrases = {}
    for candidate, idx in self.theta.items():
      phrases[candidate] = self.param_theta[idx]
    if phrases:
      phrases = pd.Series(phrases).sort_values(ascending=False).reset_index()
      phrases.columns = ['phrase', 'score']
      self._phrases = phrases

  def PGPS(self):
    """
    Algorithm 1: POS-Guided Phrasal Segmentation (PGPS)
    基于part-of-speech标签的短语分割算法

    :return: B::优化后的边界下标序列，每个条目是确定分割短语的边界下标
    """
    # hi === max_B p(w1t1 ... wntn,B|Q,theta,delta)
    # default -1e100：在数空间中用这个数代表常数中的零
    # exp(-1e100) ≈ 0

    tiny_log_value = -1e100

    h = collections.defaultdict(lambda: tiny_log_value)

    # 初始化句首边界的概率
    # 在对数空间中用0.代表常数中的 1
    # log(1) = 0
    h[1] = 0.

    # g <- {右边界: 左边界}
    g = {}

    bar = tqdm.tqdm(range(1, len(self.C) + 1), ascii=True)
    for i in bar:
      bar.set_description('PGPS')

      for j in range(i + 1, min(i + self.length_threshold + 1,
                                len(self.C) - 1)):
        if not self.Q.startswith(self.C.word(i, j)):
          h[j] = h[i]
          g[j] = i
          break

        # 获取参数
        T_values = self.pos_quality_scores(i, j)
        theta_values = self.length_quality_scores(i, j)
        Q_values = self.freq_quality_scores(i, j)

        # 计算右边界下标的出现概率
        # 概率乘积很容易小到超过正常浮点数的范围，
        # 用乘法转成对数加法，让计算机能够有效存储的概率乘积
        # 比较大小时，用log加法替代乘法
        # log(a)+log(b)+log(c)+log(d) <-- a*b*c*d

        # p(bi+1, dw[bi,bi+1)c|bi, t) =
        # T(t[bi,bi+1))θw[bi,bi+1)Q(w[bi,bi+1))
        # p ∈ 实数空间
        p = np.append(
          T_values[:-1],  # 短语pos标签内部概率
          ((1 - T_values[-1]),  # 短语pos标签外部概率
           theta_values,
           Q_values))

        # 通过 mask 的方法将 log(0) 替换成 -1e100
        log_p = np.log(p, out=np.zeros_like(p), where=(p != 0))

        # TODO: 解决 log(零）问题
        # log_p[log_p == 0] = tiny_log_value
        sum_log_p = log_p.view(np.float64).sum()

        if h[i] + sum_log_p > h[j]:
          h[j] = h[i] + sum_log_p
          g[j] = i

    j = len(self.C) + 1
    m = 0
    b = {}

    while j > 1:
      m += 1
      b[m] = j
      try:
        j = g[j]
      except KeyError:
        j = j - 1

    B = [1]

    for i in reversed(range(1, m)):
      B.append(b[i])

    return B

  def init_parameters(self, memory_size_mb):
    """参数初始化"""

    # 1.theta参数初始化

    # 遍历所有可能的候选短语，统计词频
    counter = bounter.bounter(size_mb=memory_size_mb)
    for i in range(1, len(self.C) + 1):
      for j in range(i + 1, min(i + self.length_threshold + 1,
                                len(self.C) - 1)):
        if not self.Q.startswith(self.C.word(i, j)):
          break
        counter.update((self.C.word(i, j),))

    # 词频分布
    distr = np.array([i for i in counter.values()])
    _mean, _std = distr.mean(), distr.std()
    norm = truncnorm((0. - _mean) / _std, (1. - _mean) / _std,
                     loc=_mean,
                     scale=_std)

    # θ <- {短语 <- 短语下标}
    theta = bounter.bounter(size_mb=memory_size_mb)

    idx_theta = 0  # 从零计数
    for candidate in counter.keys():
      theta[candidate] = idx_theta
      idx_theta += 1
    del counter

    # 按正态分布随机初始化 θ
    param_theta = norm.rvs(size=idx_theta)

    # 2.delta参数初始化

    # 获取所有可能的pos标签二元组，
    delta = self.Tx_Ty()
    # 按均匀分布随机初始化 δ
    param_delta = np.random.uniform(size=len(delta))

    self._delta = delta
    self._theta = theta
    self._param_delta = param_delta
    self._param_theta = param_theta

    # 检查参数收敛情况
    self._trace_param_delta.append(
      np.less(param_delta, np.ones(param_delta.size)).astype(int).sum()
    )
    self._trace_param_theta.append(
      np.less(param_theta, np.ones(param_theta.size)).astype(int).sum()
    )

  def update_theta(self, B, memory_size_mb):
    assert (isinstance(B, list))
    assert len(B) > 1

    pre_param = self.param_theta

    m = len(B)

    numerator = bounter.bounter(size_mb=memory_size_mb // 2)
    denominator = collections.defaultdict(int)

    bar = tqdm.tqdm(range(m - 1), ascii=True)

    for i in bar:
      bar.set_description('theta 1/2')
      numerator.increment(self.C.word(B[i], B[i + 1]))
      denominator[B[i + 1] - B[i]] += 1

    bar = tqdm.tqdm(self.theta.items(),
                    total=len(self.theta),
                    ascii=True)

    for candidate, idx in bar:
      bar.set_description('theta 2/2')

      if numerator[candidate] != 0:
        u_len = candidate.count(' ') + 1

        new_theta_value = numerator[candidate] / denominator[u_len]
        self.param_theta[idx] = new_theta_value
      else:
        self.param_theta[idx] = 0.

    self._trace_param_theta.append(
      np.less(self.param_theta, pre_param).astype(int).sum()
    )

    del pre_param

  def update_delta(self, B, memory_size_mb):
    assert (isinstance(B, list))
    assert len(B) > 1

    pre_param = self.param_delta

    m = len(B)
    n = len(self.C)

    TxTy_numerator = bounter.bounter(size_mb=memory_size_mb // 2)
    TxTy_denominator = bounter.bounter(size_mb=memory_size_mb // 2)

    bar = tqdm.tqdm(range(1, m - 1), ascii=True)

    for i in bar:
      bar.set_description('delta 1/3')

      for j in range(B[i], B[i + 1] - 1):
        TxTy_numerator.increment(self.C.tag(j, j + 2))

    bar = tqdm.tqdm(range(1, n), ascii=True)

    for i in bar:
      bar.set_description('delta 2/3')
      TxTy_denominator.increment(self.C.tag(i, i + 2))

    bar = tqdm.tqdm(self.delta.items(),
                    total=len(self.delta),
                    ascii=True)

    for txty, idx in bar:
      bar.set_description('delta 3/3')

      if TxTy_numerator[txty] != 0:
        print(txty)
        new_delta_value = TxTy_numerator[txty] / TxTy_denominator[txty]
        self.param_delta[idx] = new_delta_value
      else:
        self.param_delta[idx] = 0.

    self._trace_param_delta.append(
      np.less(self.param_delta, pre_param).astype(int).sum()
    )

    del pre_param

  def length_quality_scores(self, start_idx, end_idx):
    """获取指定的 theta[u] 参数"""
    # 初始化时从1开始计数，np.array是从0开始计数，因此 u_idx=idx - 1
    u_idx = self.theta[self.C.word(start=start_idx, stop=end_idx)]
    param_val = self.param_theta[u_idx]
    return param_val

  def pos_quality_scores(self, start_idx, end_idx):
    """获取T[u] 参数"""

    # 候选短语与右邻单词有2种组合情况
    #
    # 1.候选短语和右邻单词都存在：(t1 t2 t3), t4
    #   内部概率：(t1 t2), (t2 t3)
    #   外部概率：(t3 t4)
    #
    # 2.候选短语只有一个单词的情况：(t1), t2
    #   内部概率：(t1 t1) = 1.0
    #   外部概率：(t1 t2)

    # 有两部分组成：
    # T_values[:-1]: 内部概率
    # T_values[-1]: 外部概率
    delta_param_vals = []

    if end_idx < len(self.C):

      if end_idx - start_idx > 1:

        # 多词词组的情况，遍历内部二元词组和右邻词组
        for k in range(start_idx, end_idx + 1):
          TxTy = self.C.tag(k, k + 2)
          TxTy_idx = self.delta[TxTy]
          param_val = self.param_delta[TxTy_idx]
          delta_param_vals.append(param_val)
      else:

        # 单词词组的情况，内部完整性概率为100%
        TxTy = self.C.tag(start_idx, end_idx + 1)
        TxTy_idx = self.delta[TxTy]
        param_val = self.param_delta[TxTy_idx]
        delta_param_vals.append(1.)
        delta_param_vals.append(param_val)

    if delta_param_vals:
      return np.array(delta_param_vals)

  def freq_quality_scores(self, start_idx, end_idx, epsilon=1e-3):
    """initialised by AutoPhrase.features"""
    try:
      q = self.Q[self.C.word(start_idx, end_idx)]
    except KeyError:
      q = epsilon
    return q

  def Tx_Ty(self):
    """获取所有可能的tx ty
    part-of-speech by hanlp"""
    tags = []
    f = open(self.FNAME_POSTAG)
    while True:
      line = f.readline()
      if not line:
        break
      if line.startswith('#'):
        continue

      line = line.rstrip()
      tags.append(line)

    if tags:
      TxTy_dict = {}
      for idx, (tx, ty) in enumerate(itertools.product(tags, tags)):
        TxTy_dict['{} {}'.format(tx, ty)] = idx
      return TxTy_dict
    else:
      return {}


class Omega:

  def __init__(self, corpus):
    """语料库转为入参形式 Ω = Ω1 Ω2 ... Ωn
    其中 Ω_i = (word_i, tag_i)
    输入 Ω[2:5] 将返回 (Ω2 Ω3 Ω4) """
    self._unigrams = list(Ngram(corpus, n=1, ignore_unigram=False))
    logging.debug('Omega initials.')

  def __len__(self):
    return len(self._unigrams)

  def __getitem__(self, sliced):
    return self._unigrams[sliced]

  def word(self, start, stop=None):
    _words = []
    for term in self._unigrams[start:stop]:
      _words.append(str(term.word))
      if stop is None:
        break
    return ' '.join(_words)

  def tag(self, start, stop=None):
    _tags = []
    for term in self._unigrams[start:stop]:
      _tags.append(str(term.tag))
      if stop is None:
        break
    return ' '.join(_tags)


class Quality:

  def __init__(self, fname):
    """
    用trie树构造短语质量查询表

    :param fname: 从AutoPhrase.features 生成的质量表（候选词必须包含unigram）
    :return: Q::基于词频获取的短语质量表 Q = {phrase: quality_score}
    """
    try:
      Q = pd.read_excel(fname)
      Q['candidate'] = Q['candidate'].astype(str)
      Q = Q.set_index('candidate').to_dict()['scores']
    except KeyError:
      raise KeyError('no column named scores in {}'.format(fname))

    # 返回trie树和词性字典
    self.root = {}
    _end = '_e_'

    # 构建trie树
    for term, scores in Q.items():
      current_dict = self.root

      # 字粒度转成单词粒度
      words = term.split(' ')

      for word in words:
        current_dict = current_dict.setdefault(word, {})
      current_dict[_end] = scores

  def startswith(self, term):
    """判断短语质量表中是否存在 term"""
    _end = '_e_'
    current_dict = self.root

    words = term.split(' ')
    for word in words:
      if word in current_dict:
        return True
      else:
        return False

  def __getitem__(self, term):
    _end = '_e_'
    current_dict = self.root

    words = str(term).split(' ')
    for word in words:
      if word in current_dict:
        current_dict = current_dict[word]
      else:
        raise KeyError()
    else:
      if _end in current_dict:
        return current_dict[_end]
      else:
        raise KeyError()


class Ngram:
  Term = collections.namedtuple('Term', 'word tag')

  def __init__(self, corpus, n, ignore_unigram):
    """将语料库转换成N-gram序列"""
    self.corpus = corpus
    self.n = n
    self.ignore_unigram = ignore_unigram

  def __iter__(self):
    for doc in self.corpus:
      if not isinstance(doc, Sentences):
        err = 'doc must be a Sentence object. but got: {}'.format(type(doc))
        raise TypeError(err)

      for sent in doc:
        # sent = (word1, tag1) ... (wordN, tagN)
        sent_words = [word for word, tag in sent]
        sent_tags = [tag for word, tag in sent]

        # 获取单词、pos标签两个序列的二元组
        ngram_words = ngram(sent_words, self.n)
        ngram_tags = ngram(sent_tags, self.n)

        for word, tag in zip(ngram_words, ngram_tags):
          if self.ignore_unigram and ' ' not in word:
            continue

          term = self.Term(word=word, tag=tag)
          yield term


class Corpus:

  def __init__(self, fname):
    self.fname = fname

  def __iter__(self):
    dataframe = pd.read_excel(self.fname)
    for _, r in dataframe.iterrows():
      cont = r['文本内容']
      sentences = Sentences(cont)
      yield sentences


class Sentences:

  def __init__(self, strings, eos_placement='※'):
    """将document转换成a sequence of sentence"""
    self.strings = strings
    self.eos_placement = eos_placement

  def __iter__(self):
    """make each sentence a new line"""
    normed_sent = preprocess(self.strings)
    for sent in split_iter(normed_sent, self.eos_placement):
      sent = ''.join(sent)
      if sent:
        yield list(tokenize(sent))


def tokenize(strings):
  """中文分词 + 词性标注"""
  for term in HanLP.segment(strings):
    yield term.word, str(term.nature)


def decreasing(seq):
  return all(x > y for x, y in zip(seq, seq[1:]))


if __name__ == '__main__':
  # # test AutoPhrase.train on whole datasets
  # corpus_instance = Corpus('test.xlsx')
  # quality_instance = Quality('QualityPhrase（新能源2017年）.xlsx')
  # omega_instance = Omega(corpus_instance)
  # ap = AutoPhrase(omega=omega_instance,
  #                 quality=quality_instance,
  #                 length_threshold=6)
  # ap.train(memory_size_mb=10240)
  # b_list = ap.PGPS()
  # ipsh()

  # # test decreasing
  # seq1 = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
  # assert (decreasing(seq1) == True)
  #
  # seq2 = [9, 8, 7, 6, 5, 4, 3, 3, 3, 3]
  # assert (decreasing(seq2) == False)
  #
  # seq3 = [9, 8, 8]
  # assert (decreasing(seq3) == False)

  # test AutoPhrase.train
  fname_corpus = 'data/test.xlsx'
  corpus_instance = Corpus(fname_corpus)
  quality_instance = Quality('data/QualityPhrase（新能源2017年）.xlsx')
  omega_instance = Omega(corpus_instance)
  ap = AutoPhrase(omega=omega_instance,
                  quality=quality_instance,
                  length_threshold=6)
  ap.train(memory_size_mb=512)
  b_list = ap.PGPS()
  ap.phrases.to_excel('res.xlsx', index=False)
  ipsh()
  #
  # with open('output-b_list.txt', 'w') as fw:
  #   for line in b_list:
  #     fw.write('{}\n'.format(line))

  # test AutoPhrase.init_parameters
  # test AutoPhrase.PGPS
  # test AutoPhrase.update_delta
  # fname_corpus = 'data/test.xlsx'
  # corpus_instance = Corpus(fname_corpus)
  # quality_instance = Quality('data/QualityPhrase（新能源2017年）.xlsx')
  # omega_instance = Omega(corpus_instance)
  # ap = AutoPhrase(omega=omega_instance,
  #                 quality=quality_instance,
  #                 length_threshold=6)
  # ap.init_parameters(memory_size_mb=128)
  # B_lst = ap.PGPS()
  #
  # ap.update_delta(B_1st_round, memory_size_mb=128)
  # B_2nd_round = ap.PGPS()
  # ap.update_delta(B_2nd_round, memory_size_mb=128)
  # B_3rd_round = ap.PGPS()
  #
  # with open('output-B_lst.txt', 'w') as fw:
  #   for line in B_lst:
  #     fw.write('{}\n'.format(line))

  # with open('output-B_2nd_round.txt', 'w') as fw:
  #   for line in B_2nd_round:
  #     fw.write('{}\n'.format(line))
  #
  # with open('output-B_3rd_round.txt', 'w') as fw:
  #   for line in B_3rd_round:
  #     fw.write('{}\n'.format(line))

  # # test AutoPhrase.PGPS
  # fname_corpus = 'data/test.xlsx'
  # corpus_instance = Corpus(fname_corpus)
  # quality_instance = Quality('data/QualityPhrase（新能源2017年）.xlsx')
  # omega_instance = Omega(corpus_instance)
  # ap = AutoPhrase(omega=omega_instance,
  #                 quality=quality_instance,
  #                 length_threshold=6)
  # ap.init_parameters(memory_size_mb=128)
  # res = ap.PGPS()
  #
  # with open('output-G.txt', 'w') as fw:
  #   for line in res:
  #     fw.write('{}\n'.format(line))

  # # test Ngram
  # corpus = Corpus('data/test.xlsx')
  # bigram = Ngram(corpus, n=2, ignore_unigram=True)
  # for i in bigram:
  #   print(i)
  #
  #
  # # test Quality implemented by trie tree
  # trie = Quality('data/QualityPhrase（新能源2017年）.xlsx')
  # # assert '插 电 式 混合' in trie
  # # assert '插 电 式 混合 动力 城市' in trie
  # # assert '插 电 式 混合 动' not in trie
  # print(trie['插 电 式 混合'])
  # print(trie['插 电 式 混合 动力 城市'])
  # print(trie.startswith('插'))
  # ipsh()
  #
  #
  # # test Omega
  # corpus = Corpus('data/test.xlsx')
  # omega = Omega(corpus)
  # assert omega.word(0, 15) == ' '.join(t.word for t in omega[:15])
  # assert omega.tag(0, 15) == ' '.join(t.tag for t in omega[:15])
