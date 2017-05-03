# Data Store Class: Reads dataset from file and feeds mini batches
# on request

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.platform import gfile

import flags
import mapper
import reverse_sentence as rvs

FLAGS = flags.FLAGS

buckets = [64, 96, 128, 160, 256]

class DataStoreConfig(object):
  num_extra_ids       = 2 # one for SPACE and one for GO during decoding
  data_dir            = FLAGS.data_dir
  valid_file          = FLAGS.valid_file
  test_file           = FLAGS.test_file
  train_file          = FLAGS.train_file
  max_train_data_size = FLAGS.max_train_data_size
  mini_batch_size     = FLAGS.mini_batch_size
  # Use buckets in sequence to sequence and pad to closest one
  # for efficiency. Characters in a sentences that are beyond
  # last buckets size are ignored
  buckets             = buckets
    
class DataStore(object):
  def __init__(self, dsc, mapper):
    self.mapper    = mapper
    self.ds_config = dsc
    self.buckets   = dsc.buckets
    self._read_datasets(dsc.data_dir,
      dsc.valid_file, dsc.test_file,
      dsc.train_file, dsc.max_train_data_size)

  def sample_bucket(self, bucket_scale):
    # Choose bucket according to data distribution using Smirnov Transformation
    # Generate random number in [0, 1] range and use corresponding
    # interval in train_bucket_scale
    rnd_01 = np.random.random_sample()
    for idx in xrange(len(bucket_scale)):
      if bucket_scale[idx] > rnd_01:
        return idx

  def get_mini_batch(self, data_set, bucket_id):
    """ Get a random batch of data from bucket_id
    
    Data is a list of length-major cases. We convert to batch-major vectors so
    that we can feed it to sequence to sequence computation graph.
    
    Args:
    data: List of list-pairs of input data and output labels
    bucket_id: integer (id) specifies bucket_index we should use
    encoder_input: A list of tokens. If None one list if picked in random

    Returns:
    Triple (encoder inputs, decoder inputs, target weights)
    """
    encoder_inputs, decoder_inputs = [], []
    # Get a random batch of encoder followed by EOS followed by PADs
    # decoder is just reversed encoder prepended by PADs
    for _ in xrange(self.ds_config.mini_batch_size):
      encoder_input = random.choice(data_set[bucket_id])
      decoder_input = list(reversed(encoder_input))
      encoder_data, decoder_data = self._get_data_row(
        encoder_input, decoder_input, bucket_id)
      encoder_inputs.append(encoder_data)
      decoder_inputs.append(decoder_data)

    return self._prep_batch_row_data(encoder_inputs, decoder_inputs, bucket_id)

  def create_bucket_scale(self, dataset):
    bucket_sizes = [len(dataset[idx])
                    for idx in xrange(len(self.buckets))]
    tot_size = sum(bucket_sizes)
    total_size = float(tot_size)
    bucket_scale = [sum(bucket_sizes[:idx+1])/total_size
                    for idx in xrange(len(bucket_sizes))]
    return bucket_scale, tot_size

  def bucketize(self, source_ids):
    len_ids = len(source_ids)
    # identify bucket_id
    for bucket_id, bucket_size in enumerate(self.buckets):
      if bucket_size - (len_ids + self.ds_config.num_extra_ids) >= 0:
        return bucket_id, source_ids

    # string exceeds max bucket size
    max_bucket_id   = len(self.buckets)-1
    max_bucket_size = self.buckets[max_bucket_id]
    max_len_ids     = max_bucket_size - self.ds_config.num_extra_ids
    source_ids      = source_ids[:max_len_ids]
    assert len(source_ids) < self.buckets[max_bucket_id]

    return max_bucket_id, source_ids  

  def get_source_ids(self, str):
    return [self.mapper.char2id.get(ch, self.mapper._UNK_ID)
            for ch in str]
  
  def _read_datasets(self, data_dir,
                     valid_file, test_file,
                     train_file, max_train_size):
    """
    Raises:
      ValueError: if data_dir, valid_file, or train_file does not exist
    """
    if not os.path.exists(data_dir):
      raise ValueError("Data Directory {} does not exist".format(data_dir))
    
    va_file   = os.path.join(data_dir, valid_file)
    if not gfile.Exists(va_file):
      raise ValueError("Valid Data File {} does not exist".format(va_file))
    te_file   = os.path.join(data_dir, test_file)
    if not gfile.Exists(te_file):
      raise ValueError("Test Data File {} does not exist".format(te_file))
    tr_file   = os.path.join(data_dir, train_file)
    if not gfile.Exists(tr_file):
      raise ValueError("Train Data File {} does not exist".format(tr_file))

    self.valid_dataset = self._read_data(va_file, 0)
    self.test_dataset  = self._read_data(te_file, 0)
    self.train_dataset = self._read_data(tr_file, max_train_size)

    return

  def _get_data_row(self, encoder_input, decoder_input, bucket_id):
    """ An input row data is prepped to seq2seq format
    Args:
    encoder_input: raw encoder input sequence
    decoder_input: raw decoder input sequence
    bucket_id:      bucket idx where data row belongs
    Returns:   
    Encoder & Decoder data in time row major form 
    """
    bucket_size = self.buckets[bucket_id]

    inp_pad_list  = [self.mapper._PAD_ID]*(
      bucket_size - len(encoder_input))
    label_pad_list= [self.mapper._PAD_ID]*(
      bucket_size - (len(decoder_input)+self.ds_config.num_extra_ids))

    # Input is padded in the beginning of sentence to facilitate
    # learning by keeping characters close to output
    encoder_data = inp_pad_list + encoder_input 
    decoder_data = ( [self.mapper._GO_ID] + decoder_input +
                [self.mapper._EOS_ID] + label_pad_list)
    assert(len(encoder_data) == len(decoder_data))
    assert(len(encoder_data) == bucket_size)

    return encoder_data, decoder_data

  def _prep_batch_row_data(self, encoder_inputs, decoder_inputs, bucket_id):
    bucket_size = self.buckets[bucket_id]
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Create batch-major vectors
    for time_idx in xrange(bucket_size):
      batch_encoder_inputs.append(
        np.array([encoder_inputs[row_idx][time_idx]
                  for row_idx in xrange(self.ds_config.mini_batch_size)],
                 dtype=np.int32))
      batch_decoder_inputs.append(
        np.array([decoder_inputs[row_idx][time_idx]
                  for row_idx in xrange(self.ds_config.mini_batch_size)],
                 dtype=np.int32))
  
    for time_idx in xrange(bucket_size-1):
      batch_weight = np.ones(
        self.ds_config.mini_batch_size,dtype=np.float32)
      batch_weight[batch_decoder_inputs[time_idx + 1]==self.mapper._PAD_ID] = 0.0
      batch_weights.append(batch_weight)
    batch_weights.append(
      np.zeros(self.ds_config.mini_batch_size,dtype=np.float32))

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

  def _read_data(self, filename, max_size=None):
    """ Read data from file, index data into token_ids, and put into buckets

    Args:
    filename: file from where to read text
    max_size: maximum number of lines to read. rest are ignored.

    Returns:
    data_set: a list of len(buckets); data_set[n] contains a list
      of source, target pairs read from file that fit into n-th
      bucket i.e. len(source) < _bucket[n] and remaining characters
      padded. len(target) = len(source) + 1 due to additional GO symbol.
    """
    if not tf.gfile.Exists(filename):
      raise ValueError("file %s not found.", filename)

    data_set = [[] for _ in self.buckets]

    print("Reading data from file {}".format(filename))
    with tf.gfile.GFile(filename, "r") as f:
      counter = 0

      for line in f:
        if max_size and counter >= max_size:
          break;

        counter += 1;
        if counter % 10000 == 0:
          print("  reading data line# {}: {}".format(counter, line))
        
        line = tf.compat.as_str(line.strip().
                                replace("<unk>", self.mapper._UNK))
        bucket_id, source_ids = self._process_line(line)
        if bucket_id == None:
          continue
        data_set[bucket_id].append(source_ids)

    return data_set    

  def _process_line(self, line):
    len_line = len(line)
    if len_line == 0:
      return None, None

    return self.bucketize(self.get_source_ids(line))
