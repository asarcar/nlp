# Main Code and Global Configuration to run the model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy             as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow        as tf
import sys
import time

import flags
import mapper
import data_store
import reverse_model

FLAGS = flags.FLAGS

def main(_):
  if FLAGS.self_test:
    print("Self_test for reverse sentence model.")
    if not FLAGS.decode:
      self_test_train()
      return

    self_test_decode()
    return
  
  if FLAGS.decode:
    decode()
    return

  train()


def train():
  """ Train a model to translate a sentence and spit out its reverse. """
  m   = mapper.Mapper()
  dsc = data_store.DataStoreConfig()
  ds  = data_store.DataStore(dsc, m)
  rmc = reverse_model.ReverseModelConfig()
  rm  = reverse_model.ReverseModel(rmc, dsc, m)

  epoch_losses = [float('inf') for _ in xrange(FLAGS.anneal_epochs)]
  val_losses   = [float('inf')]
  best_val     = float('inf')
  best_idx     = 0
  with tf.Session() as sess:
    rm.init_params(sess)
    # This is the training loop.
    for epoch in xrange(FLAGS.max_epochs):
      start_time   = time.time() 
      epoch_loss, val_loss = run_epoch(sess, rm, ds)
      epoch_time = (time.time()-start_time)

      epoch_losses.append(epoch_loss)      
      val_losses.append(val_loss)

      # Decrease learning if no improvements over last 'rmc.anneal_epochs'
      if epoch_loss >= max(epoch_losses[-FLAGS.anneal_epochs:]):
        prev_lr = rm.lr.eval()
        sess.run(rm.lr_decay_op)
        print("Epoch {} Time {:.2f} Loss {:.4f} Perplexity {:.4f}".
              format(epoch, epoch_time, epoch_loss, perplexity(epoch_loss)))
        print("    Global Step {} Learning Reduced from {:.4f} to {:.4f}".
              format(rm.global_step.eval(), prev_lr, rm.lr.eval()))

      # Save Model if validation loss has improved
      best_val = min(val_losses)
      best_idx = val_losses.index(best_val)
      max_idx  = len(val_losses)-1
      if max_idx == best_idx:
        rm.save_params(sess)
        
      # Early Stopping: Model has not improved for some time terminate learning
      if max_idx - best_idx >= FLAGS.early_stop:
        print("Training Stopped at epoch {}: Best Validation {:.4f} at epoch {}".
              format(epoch, best_val, best_idx-1))
        break;
      
    print("\n\nTraining stopped after epoch {}: ".format(epoch))
    print("    Best validation loss at epoch {} is {:.4f} Perplexity {:.4f}".
          format(best_idx-1, best_val, perplexity(best_val)))
    test_loss = xtract_loss(sess, ds.test_dataset, rm, ds)
    print("Test Loss {:.4f} Perplexity {:.4f}". 
          format(test_loss, perplexity(test_loss)))

def decode():
  m   = mapper.Mapper()
  dsc = data_store.DataStoreConfig()
  dsc.mini_batch_size = 1
  ds  = data_store.DataStore(dsc, m)
  rmc = reverse_model.ReverseModelConfig()
  rmc.forward_only=True
  rm  = reverse_model.ReverseModel(rmc, dsc, m)
  
  # Decode from standard input.
  sys.stdout.write("> ")
  sys.stdout.flush()
  sentence = sys.stdin.readline().strip()
  while sentence:
    token_ids = ds.get_source_ids(sentence)
    bucket_id, token_ids = ds.bucketize(token_ids)
    print("token_ids: len {} ids {} bucket_id {}".
          format(len(token_ids), token_ids, bucket_id))
    
    with tf.Session() as sess:
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = ds.get_mini_batch(
        {bucket_id: [token_ids]}, bucket_id)
      pad_str = "".join([m.id2char[inp[0]] for inp in encoder_inputs])
      rm.init_params(sess)
    
      # Get output logits for the sentence.
      _, logits, attns = rm.output_data(
        sess, encoder_inputs, decoder_inputs, target_weights, bucket_id)
      
    display_result(sentence, pad_str, m, logits, attns)
      
    print("> ", end="")
    sys.stdout.flush()
    sentence = sys.stdin.readline().strip()

def display_result(inp_str, pad_str, m, logits, attns):
  # Greedy decoder - no beam search - argmaxes of logit at time slot(t)
  # was input to time slot(t+1)
  outputs = [int(np.argmax(logit, axis=1)) for logit in logits]
  assert len(attns) == len(outputs)

  # If there is an EOS symbol in outputs, cut them at that point.
  if m._EOS_ID in outputs:
    outputs = outputs[:outputs.index(m._EOS_ID)]
  # Print out sentence corresponding to outputs.
  rev_str = "".join([str(m.id2char[id]) for id in outputs])
  print("Reverse of input_str \n'{}' is \n'{}'".
        format(inp_str,rev_str))

  lenstr    = len(rev_str)
  lenpadstr = len(pad_str)
  
  attns = attns[:lenstr]
  top_attns_list = [(-attn[0]).argsort()[:2] for attn in attns]
  print("Top 2 attns: rev inp\n'{}:'\n'{}'\n'{}'".
        format(pad_str,
               "".join([pad_str[idxs[0]] for idxs in top_attns_list]),
               "".join([pad_str[idxs[1]] for idxs in top_attns_list])))

  # display attn_mat in grayscale along with input
  attn_mat = (255*np.concatenate(attns, axis=0)).astype(int)
  assert (attn_mat.ndim == 2 and attn_mat.shape[0] == lenstr and
          attn_mat.shape[1] == lenpadstr)
  plt.xticks(np.arange(lenpadstr),list(pad_str))
  plt.yticks(np.arange(lenstr), list(rev_str), rotation='vertical')
  plt.gray()
  plt.imshow(attn_mat)
  img_filename = FLAGS.train_dir + '/' + inp_str + '.jpg'
  plt.savefig(img_filename)
    
  # Greedy decoder - no beam search - argmaxes of logit at time slot(t)
  # was input to time slot(t+1)
  logits = logits[:lenstr]
  top_chars_list = [(-logit[0]).argsort()[:2] for logit in logits]
  print("Top two chars for input \n'{}' are \n'{}' and \n'{}'".format(
    inp_str,
    "".join([m.id2char[idxs[0]] for idxs in top_chars_list]),
    "".join([m.id2char[idxs[1]] for idxs in top_chars_list])))
    
  
def run_epoch(sess, rm, ds):
  """ Run through a single epoch of mini batches over data
  Args:
  rm:   reverse_sentence model object
  ds:   data_store object

  Returns:
  epoch_loss: average loss over all mini batches over training data
  val_loss:   average loss over all mini batches over validation data
  """
  avg_loss = 0.0
  train_losses = []
  bucket_scale, total_size = ds.create_bucket_scale(ds.train_dataset)
  num_tr_steps = total_size//ds.ds_config.mini_batch_size
  for step in xrange(num_tr_steps):
    bucket_id = ds.sample_bucket(bucket_scale)
    enc_inputs, dec_inputs, tar_wts = ds.get_mini_batch(
      ds.train_dataset, bucket_id)
    train_loss = rm.train_data(sess, enc_inputs, dec_inputs,
                               tar_wts, bucket_id)
    avg_loss += train_loss/FLAGS.steps_per_loss_report
    if (step+1)%FLAGS.steps_per_loss_report == 0:
      print("Iteration {:5d}: avg_loss {:.4f}".format(step, avg_loss))
      avg_loss = 0.0
    train_losses.append(train_loss)

  return (np.mean(train_losses),
          xtract_loss(sess, ds.valid_dataset, rm, ds))

def xtract_loss(sess, dataset, rm, ds):
  losses = [] 
  bucket_scale, total_size = ds.create_bucket_scale(dataset)
  num_va_steps = total_size//ds.ds_config.mini_batch_size
  for step in xrange(num_va_steps):
    bucket_id = ds.sample_bucket(bucket_scale)
    enc_inputs, dec_inputs, tar_wts = ds.get_mini_batch(
      dataset, bucket_id)
    loss, _, _ = rm.output_data(
      sess, enc_inputs, dec_inputs, tar_wts, bucket_id)
    losses.append(loss)
  return np.mean(losses)
    
buckets = [4,8]
def self_test_train():
  m   = mapper.Mapper()
  dsc = data_store.DataStoreConfig()
  # Use only 2 small buckets with mini_batch_size of 2 rows
  dsc.buckets         = buckets
  dsc.mini_batch_size = 2
  ds  = data_store.DataStore(dsc, m)
  rmc = reverse_model.ReverseModelConfig()
  rmc.num_hidden=8
  rmc.num_layers=2
  rm  = reverse_model.ReverseModel(rmc, dsc, m)

  print("Training Data...")

  # Fake data set for both the [3] and [6] buckets.
  dataset = [[[6, 8], [7, 9], [8, 10], [9], [10], [11]],
             [[6, 7, 8, 9, 10], [6, 9, 7, 10, 8], [8, 6, 7], [11, 9]]]
  bucket_scale, _ = ds.create_bucket_scale(dataset)

  with tf.Session() as sess:
    rm.init_params(sess)
    N=1000
    avg_loss = 0.0
    for i in xrange(N):  # Train the fake model for 5 steps.
      bucket_id = ds.sample_bucket(bucket_scale)
      encoder_inputs, decoder_inputs, target_weights = ds.get_mini_batch(
        dataset, bucket_id)
      loss = rm.train_data(sess,
                           encoder_inputs, decoder_inputs, target_weights,
                           bucket_id)
      avg_loss += loss/FLAGS.steps_per_loss_report
      if (i+1)%FLAGS.steps_per_loss_report == 0:
        print("Iteration {:5d} avg_loss {:.4f} Perplexity {:.4f}".
              format(i, avg_loss, perplexity(avg_loss)))
        avg_loss = 0.0
        rm.save_params(sess)

def self_test_decode():
  m   = mapper.Mapper()
  dsc = data_store.DataStoreConfig()
  # Use only 2 small buckets with mini_batch_size of 2 rows
  dsc.buckets         = buckets
  dsc.mini_batch_size = 1
  ds  = data_store.DataStore(dsc, m)
  rmc = reverse_model.ReverseModelConfig()
  rmc.num_hidden=8
  rmc.num_layers=1
  rm  = reverse_model.ReverseModel(rmc, dsc, m)

  print("Predicting Data...")

  # Create few tokens and identify its bucket
  orig_str  = "eee"
  token_ids = ds.get_source_ids(orig_str)
  bucket_id, token_ids = ds.bucketize(token_ids)
  
  with tf.Session() as sess:
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = ds.get_mini_batch(
          {bucket_id: [token_ids]}, bucket_id)
    print("Encoder Inputs: {}".format(encoder_inputs))
    pad_str = "".join([m.id2char[inp[0]] for inp in encoder_inputs])
    rm.init_params(sess)
    
    # Get output logits for the sentence.
    _, logits, attns = rm.output_data(
      sess, encoder_inputs, decoder_inputs, target_weights, bucket_id)

  display_result(orig_str, pad_str, m, logits, attns)

def perplexity(loss):
  if loss >= 300:
    return float("inf")

  return math.exp(float(loss))

if __name__ == "__main__":
  tf.app.run()


