from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os.path
import tensorflow as tf

from   six.moves import xrange  # pylint: disable=redefined-builtin

import flags
import mapper
import data_store
import reverse_sentence
import seq2seq
from   tensorflow.python.ops import variable_scope

FLAGS = flags.FLAGS

class ReverseModelConfig(object):
  lr           = FLAGS.learning_rate
  lr_decay     = FLAGS.learning_rate_decay_factor
  max_gradient = FLAGS.max_gradient_norm
  train_dir    = FLAGS.train_dir # dir where all params are saved
  num_hidden   = FLAGS.hidden_size
  num_layers   = FLAGS.num_layers
  keepout      = FLAGS.keepout
  forward_only = FLAGS.decode
  force_fresh  = FLAGS.force_fresh_start

class ReverseModel(object):
  """ Reverse a string sequence to sequence model
  """
  def __init__(self, rmc, dsc, m):
    """
    Raises:
      ValueError: if train_dir does not exist
    """    
    self.model_config = rmc
    if not os.path.exists(self.model_config.train_dir):
      raise ValueError("Training Directory {} does not exist".
                       format(self.model_config.train_dir))
    self.dsc = dsc
    self.num_symbols = m.num_symbols
    self.mini_batch_size = dsc.mini_batch_size
    # Buckets in legacy_seq2seq accepts tuple. We map similarly.
    self.buckets = [(i, i) for i in dsc.buckets]
    self._init_computation_graph()
    
  def init_params(self, session):
    ckpt = tf.train.get_checkpoint_state(self.model_config.train_dir)

    if ( (not self.model_config.force_fresh) and
         ckpt and
         tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
      print("Restoring model parameters from {}".
            format(ckpt.model_checkpoint_path))
      self.saver.restore(session, ckpt.model_checkpoint_path)
      return

    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    
  def save_params(self, session):
    ckpt_path = os.path.join(self.model_config.train_dir, "translate.ckpt")
    print("Saving model parameters at {}".format(ckpt_path))
    self.saver.save(session, ckpt_path, global_step=self.global_step)

  def train_data(self, session,
                 encoder_inputs, decoder_inputs,
                 target_weights, bucket_id):
    """ Train a Step of Data for the input mini-batch feed 
    and extract current loss
    Args:
    session:   Session within which the training is executed
    encoder_inputs: batch major form list data of encoder_size length
    decoder_inputs: batch major form list data of decoder_size length
    target_weights: batch major form list data of decoder_size length
    bucket_id: Index of bucket currently processed by graph

    Returns:
    Loss:      Aggregate scalar value of loss of the model
    """
    assert self.model_config.forward_only == False
    _, size = self.buckets[bucket_id]
    assert len(encoder_inputs)==len(decoder_inputs)
    assert len(decoder_inputs)==len(target_weights)

    input_feed = self._feed_input_data(encoder_inputs, decoder_inputs,
                                       target_weights, False)

    # Extract the updates and gradient norms to enforce that loss
    # is optimized whenever train_data is called.
    output_xtract = [self.updates[bucket_id],        # Updated params
                     self.gradient_norms[bucket_id], # Gradient Norm
                     self.losses[bucket_id]]         # Loss of mini-batch

    _, _, loss = session.run(output_xtract, input_feed)

    return loss

  def output_data(self, session, 
                  encoder_inputs, decoder_inputs,
                  target_weights, bucket_id):
    """ Run the model and extract output/logits for the input mini-batch feed. 
    Also extract current loss. No training/updates of params are performed.
    Args:
    encoder_inputs: batch major form list data of encoder_size length
    decoder_inputs: batch major form list data of decoder_size length
    target_weights: batch major form list data of decoder_size length
    bucket_id: Index of bucket currently processed by graph

    Returns:
    Loss:      Aggregate scalar value of loss of the model
    logits:    List of logit sequence indexed by time
    """
    # Routine called during validation/testing of a trained data
    # or while doing prediction.
    
    _, size = self.buckets[bucket_id]
    assert len(encoder_inputs)==len(decoder_inputs)
    assert len(decoder_inputs)==len(target_weights)

    input_feed = self._feed_input_data(encoder_inputs, decoder_inputs,
                                       target_weights, True)
    assert input_feed[self.keepout_ph.name] == 1.0

    output_xtract = [self.losses[bucket_id]]         # Loss of mini-batch
    for time_idx in xrange(size):
      output_xtract.append(self.logits[bucket_id][time_idx])
    for time_idx in xrange(size):
      output_xtract.append(self.attns[bucket_id][time_idx])
      
    outputs = session.run(output_xtract, input_feed)

    return outputs[0], outputs[1:size+1], outputs[size+1:]
  
  def _init_computation_graph(self):
    # Placeholders: Encoder/Decoder Inputs and Target Weights
    self._add_placeholders()
    self._add_variables()
    self._add_model()
    if not self.model_config.forward_only:
      self._add_training_ops()
    # Create Saver for all variables so that we can save/restore state
    self.saver = tf.train.Saver(tf.global_variables())    
      
  def _add_placeholders(self):
    """ Placeholders for encoder/decoder inputs and Target Weights
    Model automatically assumes target labels are Decoder Inputs
    shifted left in time by one slot. Each time slot is represented
    by 1D tensor of mini_batch_size. 
    """
    self.keepout_ph = tf.placeholder(dtype=tf.float32, name="keepout")
    self.enc_ips_ph, self.dec_ips_ph, self.tgt_wts_ph = [], [], []
    # Max Time Slots is the number of sequences hosted in last bucket
    max_time_slots = self.buckets[-1][0]
    for i in xrange(max_time_slots):
      enc_name  = "encoder{}".format(i)
      dec_name  = "decoder{}".format(i)
      wgt_name  = "weight{}".format(i)
      self.enc_ips_ph.append(
        tf.placeholder(dtype=tf.int32, shape=[None], name=enc_name))
      self.dec_ips_ph.append(
        tf.placeholder(dtype=tf.int32, shape=[None], name=dec_name))
      self.tgt_wts_ph.append(
        tf.placeholder(dtype=tf.float32, shape=[None], name=wgt_name))
    
    dec_name  = "decoder{}".format(max_time_slots)
    self.dec_ips_ph.append(
      tf.placeholder(dtype=tf.int32, shape=[None], name=dec_name))
    # Targets at decoder are inputs shifted by one
    self.targets = [self.dec_ips_ph[i+1]
                    for i in xrange(max_time_slots)]
      
  def _add_variables(self):
    """ Variable States
    """
    # Learning Related Operations/State
    self.lr          = tf.Variable(
      float(self.model_config.lr), trainable=False, name="learning-rate")
    self.global_step = tf.Variable(0,trainable=False, name="global-step")    
      
  def _add_model(self):
    """ Computation operations
    """
    # NN Computation Graph
    self._cell        = self._get_cell()
    # TBD: Per https://arxiv.org/pdf/1506.03099.pdf using prediction label
    # to compute losses during training should also be used.
    # We will look at this later.
    self.logits, self.losses, self.attns = seq2seq.model_with_buckets(
      self.enc_ips_ph, self.dec_ips_ph, self.targets, self.tgt_wts_ph,
      self.buckets, self._get_rnn_fn())    

  def _add_training_ops(self):
    """ Set up training operations: Optimization algorithm used is SGD with
    mini batches and gradient clipping to prevent exploding gradient.
    """
    assert self.model_config.forward_only == False
    self.lr_decay_op = self.lr.assign(
      self.lr*self.model_config.lr_decay)
    params = tf.trainable_variables()
    self.gradient_norms = []
    self.updates = []
    opt_algo = tf.train.GradientDescentOptimizer(self.lr)
    for b in xrange(len(self.buckets)):
      grads = tf.gradients(self.losses[b], params)
      clipped_grads, norm = tf.clip_by_global_norm(
        grads,self.model_config.max_gradient)
      self.gradient_norms.append(norm)
      self.updates.append(opt_algo.apply_gradients(
        zip(clipped_grads, params), global_step=self.global_step))

  def _get_cell(self):
    cell = tf.contrib.rnn.DropoutWrapper(
      tf.contrib.rnn.BasicLSTMCell(self.model_config.num_hidden),
      output_keep_prob=self.keepout_ph)
    if (self.model_config.num_layers == 1):
      return cell
    # MultiRNNCell
    return tf.contrib.rnn.MultiRNNCell(
      [cell for _ in range(self.model_config.num_layers)])

  def _get_rnn_fn(self):
    return lambda x,y: seq2seq.embedding_attention_seq2seq(
      x, y, self._cell,
      num_encoder_symbols=self.num_symbols,
      num_decoder_symbols=self.num_symbols,
      embedding_size=self.model_config.num_hidden,
      feed_previous=self.model_config.forward_only)
    
  def _feed_input_data(self,
                       encoder_inputs, decoder_inputs,
                       target_weights, forward_only):
    """ Feeds data for processing to the ReverseModel computation graph  
    Args:
    encoder_inputs: batch major form list data of encoder_size length
    decoder_inputs: batch major form list data of decoder_size length
    target_weights: batch major form list data of decoder_size length
    forward_only:   True if one wants to only forward propagate - set 
                    during validation, testing, or prediction.
                    False during training.

    Returns:
    feed: dictionary with all inputs (encoder, decoder, targets, weights)
    set appropriately
    """
    size = len(encoder_inputs)
    assert size == len(decoder_inputs)
    assert size == len(target_weights)

    keepout = self.model_config.keepout
    if (forward_only):
      keepout = 1.0
    input_feed = {}
    input_feed[self.keepout_ph.name] = keepout
    
    for time_idx in xrange(size):
      input_feed[self.enc_ips_ph[time_idx].name]=encoder_inputs[time_idx]
      input_feed[self.dec_ips_ph[time_idx].name]=decoder_inputs[time_idx]
      input_feed[self.tgt_wts_ph[time_idx].name]=target_weights[time_idx]

    # Targets are decoder inputs left shifted in time by one slot
    # Last slot of weights are pinned to zero.
    input_feed[self.dec_ips_ph[size].name]=np.zeros(
     [self.mini_batch_size], dtype=np.int32)

    return input_feed
