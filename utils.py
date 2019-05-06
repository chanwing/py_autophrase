from itertools import islice
import re
import os
from itertools import chain


def ngram(sentence, n):
  """ngram('ABCD', 3) --> A B C D AB BC CD DE ABC BCD"""
  # TODO: 某些业务需要将数值归一化成替换符号“<NUM>”，请加参数控制这一行为
  if not isinstance(sentence, list):
    raise TypeError('argument sentence should pass a list '
                    'object, got {}.'.format(type(sentence)))

  for n in range(1, n + 1):
    it = iter(sentence)
    result = tuple(islice(it, n))
    if len(result) == n:
      yield ' '.join(result)
    for gb in it:
      result = result[1:] + (gb,)
      yield ' '.join(result)


def preprocess(strings, word_splitter=' ',
               ignore_punc=True, eos_placement='※'):
  if not strings:
    return ''

  # <OS placement for those punctuations
  EOS_PLACEMENT = set("""。 ， ！ ？ ! ? ; ； : ： … . , " '""".split())
  # entire punctuations including both unicode and ascii series
  PUNCTUATION = one_token_a_line(fname='punctuations.txt') - EOS_PLACEMENT
  # punctuation escaped for regex
  punc_escape = '\\'.join(PUNCTUATION)
  eos_escape = '\\'.join(EOS_PLACEMENT)

  # lower ascii letters
  new_strings = strings.lower()
  # remove punctuation
  new_strings = re.sub(r'[{}]+'.format(punc_escape), word_splitter, new_strings)
  new_strings = re.sub(r'[\u3000 ]+', word_splitter, new_strings, flags=re.U)
  # insert <EOS>
  new_strings = re.sub(r'[{}]+'.format(eos_escape), eos_placement, new_strings)
  new_strings = new_strings.replace(r'\p', eos_placement)
  new_strings = new_strings.replace(r'\n', eos_placement)
  # remove extra whitespace
  new_strings = re.sub(r'[\r\t]+', word_splitter, new_strings)
  new_strings = re.sub(r' +', word_splitter, new_strings)
  if ignore_punc:
    new_strings = new_strings.replace(word_splitter, r'')
  new_strings = new_strings.strip()

  return new_strings


def one_token_a_line(fname):
  return set(open(safe_path(fname), 'r', encoding='utf8').read().strip().split('\n'))


def safe_path(path):
  return os.path.normpath(os.path.join(os.path.dirname(__file__), path))


def _takewhile(predicate, iterator, has_data):
  """
  Return successive entries from an iterable as long as the
  predicate evaluates to true for each entry.

  has_data outputs if the iterator has been consumed in the process.
  """
  for item in iterator:
    if predicate(item):
      yield item
    else:
      break
  else:
    has_data[0] = False


def isplit(iterator, value):
  """Return a lazy generator of items in an iterator, seperating by value."""
  iterator = iter(iterator)
  has_data = [True]
  while has_data[0]:
    yield _takewhile(value.__ne__, iterator, has_data)


def split_iter(iterator, sep):
  """Return a semi-lazy generator of items in an iterator, seperating by value."""
  iterator = iter(iterator)
  has_data = [True]
  while True:
    carry = []
    d = _takewhile(sep.__ne__, iterator, has_data)
    try:
      first = next(d)
    except StopIteration:
      if not has_data[0]:
        break
      yield iter([])
    else:
      yield chain([first], d, carry)
      carry.extend(d)