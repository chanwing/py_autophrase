from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import generator_stop

from collections import namedtuple
from abc import ABC, abstractmethod

try:
  from collections.abc import Mapping
except ImportError:
  Mapping = dict

from math import log
import tqdm

import bounter
import pandas as pd
import numpy as np
from sklearn import tree
import jieba
from pyhanlp import HanLP

from utils import ngram
from utils import preprocess
from utils import one_token_a_line
from utils import split_iter

import logging

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


class Rank:
  
  def __init__(self, positive_pool, negative_pool):
    if not positive_pool.columns.equals(negative_pool.columns):
      err = 'positive_pool and negative_pool dataframe must have identical columns'
      raise ValueError(err)
    self.positive_pool = positive_pool
    self.negative_pool = negative_pool
    self.forest = []
    self._ignore_words = set()
    self.features = ['idf', 'pmi', 'pkl']
    self.stopwords = one_token_a_line('stopwords.txt')
  
  @property
  def ignore_words(self):
    if self._ignore_words is None:
      err = 'attribute "ignore_words" should be used ' \
            'after calling random_forest.'
      raise AttributeError(err)
    return self._ignore_words
  
  def random_forest(self, T, K):
    for train in self._perturbed_training_set(T, K):
      clf = tree.DecisionTreeClassifier()
      clf.fit(train[self.features], train['label'])
      self.forest.append(clf)
      train_words = set(train.candidate.tolist())
      self._ignore_words.update(train_words)
    return
  
  def avg_predict_proba(self):
    if self.forest is None:
      raise ValueError('attribute "forest" has not init.')
    
    neg_to_predict = self.negative_pool.loc[
      ~(self.negative_pool.candidate.isin(self.ignore_words))]
    multi_scores = []
    
    for each_clf in self.forest:
      scores = each_clf.predict_proba(neg_to_predict[self.features])[:, 1]
      multi_scores.append(scores)
    else:
      arr = np.vstack(multi_scores)
      average_scores = arr.mean(axis=0)
      neg_to_predict = neg_to_predict.assign(scores=average_scores)
      proba_res = neg_to_predict[['candidate', 'scores']]
      proba_res.sort_values('scores', ascending=False, inplace=True)
      
      proba_res['check'] = proba_res['candidate'].apply(
        lambda s: any(sub in s for sub in self.stopwords))
      proba_res['length'] = proba_res['candidate'].apply(
        lambda s: s.count(' ') + 1)

      proba_res = proba_res.loc[proba_res['check'] == False]
      del proba_res['check']
      
      return proba_res
  
  def _perturbed_training_set(self, T, K):
    for _ in range(T):
      dataset = pd.concat([self.positive_pool.sample(K),
                           self.negative_pool.sample(K)],
                          axis=0, sort=True)
      dataset = dataset.sample(frac=1)
      yield dataset


class Feature:
  
  def __init__(self, corpus, min_freq, ngram_size):
    self.corpus = corpus
    self.min_freq = min_freq
    self.ngram_size = ngram_size
    
    self._count_T = None
    self._count_D = None
    self._candidates = None
    self._dataframe = None
    self._ft = None
    
    logging.debug('# docs to train : {:,}'.format(len(corpus)))
    logging.debug('ngram filter setting : freq={}, ngram={}'.format(min_freq, ngram_size))
  
  @property
  def candidates(self):
    if self._candidates is None:
      raise AttributeError('instance attribute '
                           '"candidates" can only access '
                           'after calling method train')
    else:
      return self._candidates
  
  @property
  def dataframe(self):
    if self._dataframe is None:
      raise AttributeError('instance attribute '
                           '"dataframe" can only access '
                           'after calling method train')
    else:
      return self._dataframe
  
  @property
  def ft(self):
    if self._ft is None:
      raise AttributeError('instance attribute '
                           '"ft" can only access '
                           'after calling method train')
    else:
      return self._ft
  
  def train(self, remove_stopwords, memory_size_mb):
    
    PMI = {}
    PKL = {}
    IDF = {}
    
    # count n-grams in corpus level(including unigram)
    self._count_T = self.ngram_counter(remove_stopwords=remove_stopwords,
                                       memory_size_mb=memory_size_mb)
    logging.debug('#total Global n-grams: {:,}'.format(self._count_T.total()))
    logging.debug('#unique Global n-grams: {:,}'.format(self._count_T.cardinality()))
    
    # candidates are filtered by Popularity (eg., min_freq > 30)
    self._candidates = set(t for t, c in self._count_T.items()
                           if c > self.min_freq and len(t) > 1)
    logging.debug('#Candidates(apply filter): {:,}'.format(len(self._candidates)))
    
    # IDF dependency: total number of docs
    total_num_doc = len(self.corpus)
    
    # IDF dependency: number of docs with term in it
    self._count_D = self.num_documents_with_ngram(memory_size_mb=memory_size_mb)
    
    # features
    for token in self._candidates:
      pmi_val = compute_pmi(token, self._count_T)
      pkl_val = compute_pkl(token, self._count_T, pmi_val)
      idf_val = compute_idf(self._count_D[token], total_num_doc)
      
      PMI[token] = pmi_val
      PKL[token] = pkl_val
      IDF[token] = idf_val
    
    # aggregate to class attributes
    pmi = pd.Series(PMI, name='pmi')
    pkl = pd.Series(PKL, name='pkl')
    idf = pd.Series(IDF, name='idf')
    logging.debug('statistic features finished.')
    
    if not pkl.index.equals(pmi.index) & \
           pkl.index.equals(idf.index):
      raise ValueError('shape and elements of these pd.Series'
                       '("pmi pkl idf") must be identical ')
    
    dataframe = pd.concat([pmi, pkl, idf], axis=1)
    dataframe.index.name = 'candidate'
    dataframe.reset_index(inplace=True)
    
    self._dataframe = dataframe
    self._ft = FeatureDict(dataframe)
    logging.debug("Training's complete")
  
  def ngram_counter(self, remove_stopwords, memory_size_mb):
    # Bounter is only a probabilistic frequency counter and
    # cannot be relied on for exact counting
    # mentioned in https://github.com/RaRe-Technologies/bounter
    count = bounter.bounter(size_mb=memory_size_mb)
    
    if remove_stopwords:
      stopwords = one_token_a_line(fname='stopwords.txt')
    else:
      stopwords = None
    
    process_bar = tqdm.tqdm(self.corpus, ascii=True)
    for doc in process_bar:
      process_bar.set_description('Counting n-grams')
      if not isinstance(doc, Sentences):
        err = 'doc must be a Sentence object. but got: {}'.format(type(doc))
        raise TypeError(err)
      
      for sent in doc:
        # filter stopwords after ngram token is formed
        # but that not getting rid of them in the raw corpus
        if remove_stopwords and stopwords is not None:
          ngram_tokens = [t for t in ngram(sent, self.ngram_size)
                          if t not in stopwords]
        else:
          ngram_tokens = ngram(sent, self.ngram_size)
        count.update(ngram_tokens)
    return count
  
  def num_documents_with_ngram(self, memory_size_mb):
    if self._candidates is None:
      raise ValueError('n-gram candidate is None, cannot count # documents '
                       'with ngram if it is not given.')
    
    # less candidates, less memory usage.
    count = bounter.bounter(size_mb=memory_size_mb // 2)
    
    process_bar = tqdm.tqdm(self.corpus, ascii=True)
    for doc in process_bar:
      process_bar.set_description('Counting doc with ngram')
      
      tokens_doc_level = set()
      if not isinstance(doc, Sentences):
        err = 'doc must be a Sentence object. but got: {}'.format(type(doc))
        raise TypeError(err)
      
      for sent in doc:
        tokens_sent_level = set(t for t in ngram(sent, self.ngram_size)
                                if t in self._candidates)
        tokens_doc_level.update(tokens_sent_level)
      
      if tokens_doc_level:
        count.update(tokens_doc_level)
    return count


class FeatureDict(Mapping):
  NamedMetrics = namedtuple('NamedMetrics', ['pmi', 'pkl', 'idf'])
  
  def __init__(self, dataframe):
    self._metrics_dict = dataframe.set_index('candidate'). \
      T.to_dict('list')
  
  def __getitem__(self, name):
    if name in self._metrics_dict:
      named_metrics = self.NamedMetrics._make(
        self._metrics_dict[name])
      return named_metrics
    else:
      raise AttributeError("No such attribute: " + name)
  
  def __iter__(self):
    for candidate, metrics in self._metrics_dict.items():
      yield candidate, self.NamedMetrics._make(metrics)
  
  def __len__(self):
    return len(self._metrics_dict)


class Sentences:
  
  def __init__(self, strings, eos_placement='※'):
    self.strings = strings
    self.eos_placement = eos_placement
  
  def __iter__(self):
    """make each sentence a new line"""
    normed_sent = preprocess(self.strings)
    for sent in split_iter(normed_sent, self.eos_placement):
      sent = ''.join(sent)
      if sent:
        yield list(term.word for term in HanLP.segment(sent))
        # yield list(jieba.cut(sent))


class Corpus(ABC):
  
  @abstractmethod
  def __len__(self):
    pass
  
  @abstractmethod
  def __iter__(self) -> Sentences:
    pass


def compute_pmi(token, count_T, epsilon=1e-5):
  """
                                    P(w[1],...,w[n+1])
  pmi(w[1],...,w[n+1]) = log ------------------------------
                                P(w[1],...,w[n])P(w[n+1])

                                   N * c(w[1],...,w[n+1])
                       ≈ log ------------------------------
                                c(w[1],...,w[n])c(w[n+1])

  where c is the count of a ngram, N is the total count of ngrams
  :param token: ngram token
  :param count_T: total count of ngram in corpus level
  :param epsilon: avoid make denominator be zero
  :return: pmi_val float
  """
  assert (len(token) > 1)  # 中文实体不考虑单个字的词汇
  N_T = count_T.total()
  numerator = N_T * count_T[token]
  token_1ton = token[:-1]
  token_last = token[-1]
  denominator = count_T[token_1ton] * count_T[token_last]
  pmi_val = log(numerator / sum((denominator, epsilon)))
  return pmi_val


def compute_pkl(token, count_T, pmi_val):
  assert (len(token) > 1)
  N_T = count_T.total()
  pkl_val = count_T[token] * pmi_val / N_T
  return pkl_val


def compute_idf(ngram_num_doc, total_num_doc):
  idf_val = log(sum([1, total_num_doc]) / sum((1, ngram_num_doc)))
  return idf_val

# TODO：增加词组过滤器 如：'^[还只会再满使约不近占只] *' '(分钟|小时|万元|时许|分钱)'