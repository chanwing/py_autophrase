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

import bounter
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from pyhanlp import HanLP

from utils import ngram
from utils import preprocess
from utils import split_iter

logging.basicConfig(format='%(asctime)s : %(message)s',
                    level=logging.DEBUG)


class AutoPhrase:
  FNAME_POSTAG = os.path.normpath(os.path.join(os.path.dirname(__file__), 'tags.txt'))

  def __init__(self, omega, quality):
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

    # 语料库
    self.C = omega
    # 短语生成模型
    self.Q = quality

    self._delta = None
    self._theta = None
    self._param_delta = None
    self._param_theta = None

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

  def train(self):
    """TODO: 基于维特比算法的训练函数"""
    pass

  def init_parameters(self, length_threshold, memory_size_mb):
    """参数初始化"""

    # 遍历所有可能的候选短语，并且统计词频，
    # 最后获取基于词频的正态分布，按正态分布
    # 随机初始化与之对应的参数

    # 统计词频
    counter = bounter.bounter(size_mb=memory_size_mb)

    for i in range(1, len(self.C) + 2):
      for j in range(i + 1, min(i + length_threshold, len(self.C) + 1)):
        if not self.Q.startswith(self.C.word(i, j)):
          break
        counter.update((self.C.word(i, j),))

    # 词频分布
    distr = np.array([i for i in counter.values()])
    left, right = 0., 1.
    _mean = distr.mean()
    _std = distr.std()
    norm = truncnorm((left - _mean) / _std, (right - _mean) / _std,
                     loc=_mean,
                     scale=_std)
    # 短语映射为下标
    theta = bounter.bounter(size_mb=memory_size_mb)
    idx_theta = 0
    for candidate in counter.keys():
      idx_theta += 1
      theta[candidate] = idx_theta
    del counter
    # 初始化
    param_theta = norm.rvs(size=idx_theta)

    # 获取所有可能的pos标签二元组，
    # 并且按均匀分布随机初始化与之对应的参数

    # 获取二元组
    delta = self.Tx_Ty()

    # 初始化
    param_delta = np.random.uniform(size=len(delta))

    self._delta = delta
    self._theta = theta
    self._param_delta = param_delta
    self._param_theta = param_theta

  def viterbi(self):
    """维特比算法"""
    pass

  def PGPS(self, length_threshold):
    """
    Algorithm 1: POS-Guided Phrasal Segmentation (PGPS)
    基于part-of-speech标签的短语分割算法

    :return: B::优化后的边界下标序列，每个条目是确定分割短语的边界下标
    """

    # hi === max_B p(w1t1 ... wntn,B|Q,theta,delta)
    # default -1e100：在数空间中用这个数代表常数中的零
    # exp(-1e100) ≈ 0

    h = collections.defaultdict(lambda: -1e100)

    # 初始化句首边界的概率
    # 在对数空间中用0.代表常数中的 1
    # log(1) = 0
    h[1] = 0.

    # g <- {右边界: 左边界}
    g = {}

    for i in range(1, len(self.C) + 2):
      for j in range(i + 1, min(i + length_threshold, len(self.C) + 1)):
        if not self.Q.startswith(self.C.word(i, j)):
          h[j] = h[i]
          g[j] = i
          break

        # 获取参数
        delta_prob = self.get_delta_param_val(start_idx=i, end_idx=j + 1)
        theta_prob = self.get_theta_param_val(start_idx=i, end_idx=j)
        quality_prob = self.get_quality_val(start_idx=i, end_idx=j)

        # # 计算右下标的出现概率
        # PGPS_core_prob = self.T(delta_prob) * theta_prob * quality_prob
        # if h[i] * PGPS_core_prob > h[j]:
        #   h[j] = h[i] * PGPS_core_prob
        #   g[j] = i

        # 计算右边界下标的出现概率
        # 概率乘积很容易小到超过正常浮点数的范围，
        # 用乘法转成对数加法，让计算机能够有效存储的概率乘积
        # 用 log(a)+log(b)+log(c)+log(d) 代表 a*b*c*d

        # p(bi+1, dw[bi,bi+1)c|bi, t) =
        # T(t[bi,bi+1))θw[bi,bi+1)Q(w[bi,bi+1))

        p = np.append(
          delta_prob[:-1],  # 短语pos标签内部概率
          ((1 - delta_prob[-1]),  # 短语pos标签外部概率
           theta_prob,
           quality_prob))

        sum_log_p = np.log(p).sum()

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

  @staticmethod
  def T(delta_prob):
    """针对pos标签序列所对应的单词，计算其具有完整语义的概率

    注意：所有概率的乘积通过log后求和替代

    :param delta_prob::np.array delta_prob包含两部分，短语内部概率
        和外部概率（array最后一个数据是外部概率，前面的都是内部概率）
    :return::float
    """
    items = np.append(delta_prob[:-1], (1 - delta_prob[-1]))
    T_val = functools.reduce(operator.mul, items, 1)
    return T_val

  def get_theta_param_val(self, start_idx, end_idx):
    """获取指定的 theta[u] 参数"""
    # 初始化时从1开始计数，np.array是从0开始计数，因此 u_idx=idx - 1
    u_idx = self.theta[self.C.word(start=start_idx, stop=end_idx)] - 1
    param_val = self.param_theta[u_idx]
    return param_val

  def get_delta_param_val(self, start_idx, end_idx):
    """获取指定的 delta[u] 参数"""

    # 候选短语与右邻单词有4种组合情况
    #
    # 1.候选短语和右邻单词都存在：(t1 t2 t3), t4
    #   内部概率：(t1 t2), (t2 t3)
    #   外部概率：(t3 t4)
    #
    # 2.候选短语只有一个单词的情况：(t1), t2
    #   内部概率：(t1 t1) = 1.0
    #   外部概率：(t1 t2)
    #
    # 3.只有候选短语没有右邻单词的情况：(t1 t2 t3), _
    #   内部概率：(t1 t2), (t2 t3)
    #   外部概率 = 0.0
    #
    # 4.候选短语只有一个单词，并且没有右邻单词：(t1), _
    #   内部概率：(t1 t1) = 1.0
    #   外部概率 = 0.0

    # 有两部分组成：
    # delta_param_vals[:-1]: 内部概率
    # delta_param_vals[-1]: 外部概率
    delta_param_vals = []

    tags = self.C.tag(start=start_idx, stop=end_idx).split()

    if end_idx < len(self.C):

      if len(tags) > 2:

        # 第1种情况
        for k in range(start_idx, end_idx):
          TxTy = self.C.tag(k, k + 2)
          TxTy_idx = self.delta[TxTy] - 1
          param_val = self.param_delta[TxTy_idx]
          delta_param_vals.append(param_val)

      else:
        # 第2种情况
        TxTy = self.C.tag(start=start_idx, stop=end_idx)
        TxTy_idx = self.delta[TxTy] - 1
        param_val = self.param_delta[TxTy_idx]
        delta_param_vals.append(1.)
        delta_param_vals.append(param_val)

    else:

      if len(tags) == 1:
        # 第4种情况
        delta_param_vals.append(1.)
        delta_param_vals.append(0.)

      else:
        # 第3种情况
        for k in range(start_idx, end_idx):
          if k == end_idx - 2:
            break
          TxTy = self.C.tag(k, k + 2)
          TxTy_idx = self.delta[TxTy] - 1
          param_val = self.param_delta[TxTy_idx]
          delta_param_vals.append(param_val)
        delta_param_vals.append(0.)

    if delta_param_vals:
      return np.array(delta_param_vals)

  def get_quality_val(self, start_idx, end_idx, epsilon=1e-5):
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


if __name__ == '__main__':
  # test AutoPhrase.PGPS
  fname_corpus = 'data/test.xlsx'
  corpus = Corpus(fname_corpus)
  quality = Quality('data/QualityPhrase（新能源2017年）.xlsx')
  omega = Omega(corpus)
  ap = AutoPhrase(omega=omega, quality=quality)
  ap.init_parameters(length_threshold=6, memory_size_mb=128)
  res = ap.PGPS(length_threshold=6)

  with open('output-G.txt', 'w') as fw:
    for i in res:
      fw.write('{}\n'.format(i))

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
