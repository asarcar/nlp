# Mapper Class: Maps characters to indices and vice versa
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string

class Mapper(object):
  # Special vocabulary symbols - we always put them at the start.
  _PAD    = u'#'
  _GO     = u'='
  _EOS    = u'$'
  _SPC    = u' '
  _UNK    = u'%'

  _PAD_ID       = 0
  _SPC_ID       = 1
  _UNK_ID       = 2
  _EOS_ID       = 3
  _GO_ID        = 4
  _NON_CHAR_IDS = 5

  def __init__(self):
    self.init_mappings()

  def ids2str(self, id_list):
    s = [self.id2char.get(id) for id in id_list]
    return "".join(s)
  
  def init_mappings(self):
    self.vocab = [self._PAD, self._GO, self._EOS, self._SPC, self._UNK]

    self.id2char = {
      self._PAD_ID: self._PAD,
      self._GO_ID:  self._GO,
      self._EOS_ID: self._EOS,
      self._SPC_ID: self._SPC,
      self._UNK_ID: self._UNK
    }

    self.char2id = {
      self._PAD: self._PAD_ID,
      self._GO : self._GO_ID,
      self._EOS: self._EOS_ID,
      self._SPC: self._SPC_ID,
      self._UNK: self._UNK_ID
    }
  
    first_letter = ord(string.ascii_lowercase[0])
    for char in string.ascii_lowercase:
      self.vocab.append(char)
      index = ord(char) - first_letter + self._NON_CHAR_IDS
      self.id2char[index] = char
      self.char2id[char]  = index
    self._vocab_size = len(self.vocab)
    print("Vocabulary Size of Data {}".format(self._vocab_size))
    assert self._vocab_size == len(self.id2char)
    assert self._vocab_size == len(self.char2id)

  @property
  def num_symbols(self):
    return len(self.vocab)
