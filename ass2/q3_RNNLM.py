import getpass
import os
import sys
import time

import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_iterator, sample

import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
from model import LanguageModel

# Let's set the parameters of our model
# http://arxiv.org/pdf/1409.2329v4.pdf shows parameters that would achieve near
# SotA numbers

class Config():
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  idx            = 0 # uniquely identifies a hyperparameter combination
  batch_size     = 128
  embed_size     = 75
  hidden_size    = 150
  num_steps      = 10
  max_epochs     = 16 # set to 16 on production
  early_stopping = 2
  dropout        = 0.95
  lr             = 0.001
  train          = True
  fname          = './results/rnnlm.weights'

class RNNLM_Model(LanguageModel):

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    self.vocab = Vocab()
    self.vocab.construct(get_ptb_dataset('train'))
    self.encoded_train = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('train')],
        dtype=np.int32)
    self.encoded_valid = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('valid')],
        dtype=np.int32)
    self.encoded_test = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('test')],
        dtype=np.int32)
    if debug:
      num_debug = 1024
      self.encoded_train = self.encoded_train[:num_debug]
      self.encoded_valid = self.encoded_valid[:num_debug]
      self.encoded_test = self.encoded_test[:num_debug]

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph.
    (When None is in a placeholder's shape, it's flexible)

    input_placeholder: Input placeholder tensor of shape
                       (None, num_steps), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, num_steps), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    ### YOUR CODE HERE
    steps = self.config.num_steps
    Dh    = self.config.hidden_size
    self.input_placeholder = tf.placeholder(
      tf.int32, [None, steps], name="input")
    # labels are used in sequence_loss to calculate
    # sparse_softmax_cross_entropy_with_logits which expects
    # integer values expressing categorical values
    self.labels_placeholder = tf.placeholder(
      tf.int32, [None, steps], name="labels")
    self.dropout_placeholder = tf.placeholder(
      tf.float32, name="dropout")
    self.initial_state_placeholder = tf.placeholder(
      tf.float32, name="initial_state")
    ### END YOUR CODE
  
  def add_embedding(self):
    """Add embedding layer.

    Hint: This layer should use the input_placeholder to index into the
          embedding.
    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: You might find tf.split, tf.squeeze useful in constructing tensor inputs
    Hint: Check the last slide from the TensorFlow lecture.
    Hint: Here are the dimensions of the variables you will need to create:

      L: (len(self.vocab), embed_size)

    Returns:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    """
    # The embedding lookup is currently only implemented for the CPU
    with tf.device('/cpu:0'):
      ### YOUR CODE HERE
      V = len(self.vocab)
      d = self.config.embed_size
      steps = self.config.num_steps
      # Trust default Xavier initializer tf.uniform_unit_scaling_initializer
      with tf.variable_scope("Embedding") as scope: 
        embeddings = tf.get_variable("EmbeddingMatrix", [V,d], trainable=True)
      # Dim(Win): [None, num_steps, d]
      win = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
      # Equivalent: [tf.squeeze(win_sp, 1) for win_sp in tf.split(1, steps, win)]
      # Output: [e1, e2, ..., en] where Dim(ei) = [None, d]
      inputs = tf.unstack(win, axis=1)
      ### END YOUR CODE
      return inputs

  def add_projection(self, rnn_outputs):
    """Adds a projection layer.

    The projection layer transforms the hidden representation to a distribution
    over the vocabulary.

    Hint: Here are the dimensions of the variables you will need to
          create 
          
          U:   (hidden_size, len(vocab))
          b_2: (len(vocab),)

    Args:
      rnn_outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, hidden_size).
    Returns:
      outputs: List of length num_steps, each a tensor of shape
               (batch_size, len(vocab)
    """
    ### YOUR CODE HERE
    V  = len(self.vocab)
    Dh = self.config.hidden_size
    # Trust default Xavier initializer tf.uniform_unit_scaling_initializer
    with tf.variable_scope("Projection") as scope:
      U  = tf.get_variable("Matrix", [Dh, V])
      b2 = tf.get_variable("Bias", [V])
    # Dim(rnn_op): [None, d]. Dim(op): None/Dh*Dh/V = None/V
    outputs = [tf.add(tf.matmul(rnn_op, U), b2) for rnn_op in rnn_outputs]
    ### END YOUR CODE
    return outputs

  def add_loss_op(self, output):
    """Adds loss ops to the computational graph.

    Hint: Use tensorflow.python.ops.seq2seq.sequence_loss to implement sequence loss. 

    Args:
      output: A tensor of shape (None, self.vocab)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    ### YOUR CODE HERE
    steps   = self.config.num_steps
    V       = len(self.vocab)
    N       = self.config.batch_size
    # Dim(output): steps*N/V
    # Dim(labels): N/steps => Map to compatible size to output
    # Sequence loss expects a list of output and labels
    # Dim(outputs[i]): batch_size/V
    output  = tf.reshape(output, [-1, steps*V])
    outputs = tf.split(1, steps, output, name="output-list")
    # Dim(labels[i]: N
    labels  = tf.unstack(self.labels_placeholder, axis=1, name="label-list")
    # Weights: Every element is equally weighted - all ones
    wt_ones = [tf.ones(N) for i in range(steps)]
    # label = tf.reshape(self.labels_placeholder, [-1])
    # wt_one = tf.ones(N*steps)
    # loss_rnn= sequence_loss([output], [label], [wt_one], name="SequenceLoss")
    loss_rnn= sequence_loss(outputs, labels, wt_ones, name="SequenceLoss")
    tf.add_to_collection("total_loss", loss_rnn)
    loss = tf.reduce_sum(tf.get_collection("total_loss"))
    ### END YOUR CODE
    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See 

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    ### YOUR CODE HERE
    with tf.variable_scope("Training") as scope:
      steps = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(self.config.lr, name="Adam")
      train_op  = optimizer.minimize(loss, global_step=steps, name="Minimize")
    ### END YOUR CODE
    return train_op
  
  def __init__(self, config):
    self.config = config
    self.load_data(debug=True)
    self.add_placeholders()
    self.inputs = self.add_embedding()
    self.rnn_outputs = self.add_model(self.inputs)
    self.outputs = self.add_projection(self.rnn_outputs)
    
    # We want to check how well we correctly predict the next word
    # We cast o to float64 as there are numerical issues at hand
    # (i.e. sum(output of softmax) = 1.00000298179 and not 1)
    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
    # Reshape the output into len(vocab) sized chunks - the -1 says as many as
    # needed to evenly divide
    if (self.config.train == True):
      output = tf.reshape(tf.concat(1, self.outputs), [-1, len(self.vocab)])
      self.calculate_loss = self.add_loss_op(output)
      self.train_step = self.add_training_op(self.calculate_loss)
      
  def add_model(self, inputs):
    """Creates the RNN LM model.

    In the space provided below, you need to implement the equations for the
    RNN LM model. Note that you may NOT use built in rnn_cell functions from
    tensorflow.

    Hint: Use a zeros tensor of shape (batch_size, hidden_size) as
          initial state for the RNN. Add this to self as instance variable

          self.initial_state
  
          (Don't change variable name)
    Hint: Add the last RNN output to self as instance variable

          self.final_state

          (Don't change variable name)
    Hint: Make sure to apply dropout to the inputs and the outputs.
    Hint: Use a variable scope (e.g. "RNN") to define RNN variables.
    Hint: Perform an explicit for-loop over inputs. You can use
          scope.reuse_variables() to ensure that the weights used at each
          iteration (each time-step) are the same. (Make sure you don't call
          this for iteration 0 though or nothing will be initialized!)
    Hint: Here are the dimensions of the various variables you will need to
          create:
      
          H: (hidden_size, hidden_size) 
          I: (embed_size, hidden_size)
          b_1: (hidden_size,)

    Args:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each of whose elements should be
               a tensor of shape (batch_size, hidden_size)
    """
    ### YOUR CODE HERE
    N       = self.config.batch_size
    Dh      = self.config.hidden_size
    d       = self.config.embed_size
    steps   = self.config.num_steps

    # Trust default Xavier initializer tf.uniform_unit_scaling_initializer
    with tf.variable_scope("RNN_LM") as scope:
      rnn_outputs = []
      h = self.initial_state_placeholder
      # Formula: h(t) = sigmoid(h(t-1)*H + e(t)*I + b_1)
      for idx, input in enumerate(inputs):
        if idx > 0:
          scope.reuse_variables()
          
        H  = tf.get_variable("HiddenXFormMatrix", [Dh, Dh])
        I  = tf.get_variable("InpWordRepMatrix", [d, Dh])
        b_1= tf.get_variable("HiddenBiasVector", [Dh])

        e  = tf.nn.dropout(input, self.dropout_placeholder)
        a  = tf.matmul(h,H) + tf.matmul(e,I) + b_1
        h  = tf.sigmoid(a)
        output = tf.nn.dropout(h, self.dropout_placeholder) # Drop Op
        rnn_outputs.append(output)
      self.final_state = h
    ### END YOUR CODE
    return rnn_outputs


  def run_epoch(self, session, data, train_op=None, verbose=10):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
    total_loss = []

    state = np.zeros([self.config.batch_size, self.config.hidden_size])
    for step, (x, y) in enumerate(
      ptb_iterator(data, config.batch_size, config.num_steps)):
      # We need to pass in the initial state and retrieve the final state to give
      # the RNN proper history
      feed = {self.input_placeholder:         x,
              self.labels_placeholder:        y,
              self.initial_state_placeholder: state,
              self.dropout_placeholder:       dp}
      loss, state, _ = session.run(
          [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
      total_loss.append(loss)
      if verbose and step % verbose == 0:
          sys.stdout.write('\r{} / {} : pp = {}'.format(
              step, total_steps, np.exp(np.mean(total_loss))))
          sys.stdout.flush()
    if verbose:
      sys.stdout.write('\r')
    return np.exp(np.mean(total_loss))

def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=100, stop_tokens=None, temp=1.0):
  """Generate text from the model.

  Hint: Create a feed-dictionary and use sess.run() to execute the model. Note
        that you will need to use model.initial_state as a key to feed_dict
  Hint: Fetch model.final_state and model.predictions[-1]. (You set
        model.final_state in add_model() and model.predictions is set in
        __init__)
  Hint: Store the outputs of running the model in local variables state and
        y_pred (used in the pre-implemented parts of this function.)

  Args:
    session: tf.Session() object
    model: Object of type RNNLM_Model
    config: A Config() object
    starting_text: Initial text passed to model.
  Returns:
    output: List of word idxs
  """
  # Imagine tokens as a batch size of one, length of len(tokens[0])
  tokens = [model.vocab.encode(word) for word in starting_text.split()]
  state = np.zeros([config.batch_size, config.hidden_size])
  input = np.zeros([config.batch_size, config.num_steps])
  for i in xrange(stop_length):
    ### YOUR CODE HERE
    input[0, 0] = tokens[i]
    # Pass in initial state and retrieve the final state
    feed = {model.input_placeholder:         input,
            model.initial_state_placeholder: state,
            model.dropout_placeholder:       1.0}
    # Pass final state of previous iteration to initial state 
    # of next iteration to build proper RNN history
    state, [y_pred] = session.run(
      [model.final_state, model.predictions], feed_dict=feed)
    ### END YOUR CODE
    next_word_idx = sample(y_pred[0], temperature=temp)
    tokens.append(next_word_idx)
    if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
      break
  output = [model.vocab.decode(word_idx) for word_idx in tokens]
  return output

def generate_sentence(session, model, config, *args, **kwargs):
  """Convenice to generate a sentence from the model."""
  return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)

def gen_RNNLM_Model(model):
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

  with tf.Session() as session:
    session.run(init)
    saver.restore(session, model.config.fname)
    starting_text = 'in palo alto'
    while starting_text:
      print ' '.join(generate_sentence(
        session, model, model.config, starting_text=starting_text, temp=1.0))
      starting_text = raw_input('> ')
      
  return

def test_RNNLM_Model(model):
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

  with tf.Session() as session:
    session.run(init)
    saver.restore(session, model.config.fname)
    test_pp = model.run_epoch(session, model.encoded_test)
    print '=-=' * 5
    print 'Test perplexity: {}'.format(test_pp)
    print '=-=' * 5

def train_RNNLM_Model(model):
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()


  best_val_pp = float('inf')
  best_val_epoch = 0

  with tf.Session() as session:  
    session.run(init)
    for epoch in xrange(model.config.max_epochs):
      print 'Epoch {}'.format(epoch)
      start = time.time()
      ###
      train_pp = model.run_epoch(
          session, model.encoded_train,
          train_op=model.train_step)
      valid_pp = model.run_epoch(session, model.encoded_valid)
      print 'Training perplexity: {}'.format(train_pp)
      print 'Validation perplexity: {}'.format(valid_pp)
      if valid_pp < best_val_pp:
        best_val_pp = valid_pp
        best_val_epoch = epoch
        saver.save(session, model.config.fname)
      if epoch - best_val_epoch > model.config.early_stopping:
        break
      print 'Total time: {}'.format(time.time() - start)
  print "Model {}: best epoch {} val pp {}".format(model.config.fname, best_val_epoch, best_val_pp)

  return best_val_pp

def train_n_test_RNNLM_Models(config):  
  model_name = "RNNLM-%d" % config.idx
  g = tf.get_default_graph()
  with g.name_scope(model_name) as scope:
    model = RNNLM_Model(config)
    best_pp = train_RNNLM_Model(model)
    test_RNNLM_Model(model)
  tf.reset_default_graph()
      
  return best_pp

def create_dir_and_get_filename(config):
  dir_name = 'lm-idx%d-batch_size%d-embed_size%d-hidden_size%d-num_steps%d-dropout%f-lr%f'% \
             (config.idx, config.batch_size, config.embed_size, config.hidden_size, config.num_steps, config.dropout, config.lr)

  result_dir = './results/' + dir_name
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)
  weight_dir = result_dir + '/weights'
  if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

  weight_file = weight_dir + '/rnnlm.weights'

  return weight_file
  
def find_best_RNNLM():
  """ Try multiple NER combinations and find the best hyperparameters.
  below combination provided best results on validation test.
  Batch Size     =      128
  Embed Size     =       75
  Hidden Size    =      150
  Num Steps      =      150
  Dropout        = 0.950000
  LearningRate   = 0.001000
  Below array of choices: first in every list best performer
  """
  batch_size_list  = np.array([128, 64])
  embed_size_list  = np.array([75, 50])
  hidden_size_list = np.array([150, 100, 67])
  num_steps_list   = np.array([10, 6]) 
  dropout_list     = np.array([0.95, 0.9])
  lr_list          = np.logspace(-3, -2, 2) # 0.001, 0.01

  config_list=[]
  index = 0
  for batch_size in batch_size_list:
    for embed_size in embed_size_list:
      for hidden_size in hidden_size_list:
        for num_steps in num_steps_list:
          for dropout in dropout_list:
            for lr in lr_list:
              config             = Config()
              config.idx         = index
              config.batch_size  = batch_size
              config.embed_size  = embed_size
              config.hidden_size = hidden_size
              config.num_steps   = num_steps
              config.dropout     = dropout
              config.lr          = lr
              config.fname       = create_dir_and_get_filename(config)
              index             += 1              
              config_list.append(config)

  result_list=[]
  for config in config_list:
    best_pp = train_n_test_RNNLM_Models(config);
    result_list.append(best_pp)

  val, idx = min((v,i) for (i, v) in enumerate(result_list))
  config   = config_list[idx]

  print "Best Hyperparameters"
  print "--------------------"
  print "ParameterIndex = %8d"   % config.idx
  print "Batch Size     = %8d"   % config.batch_size
  print "Embed Size     = %8d"   % config.embed_size
  print "Hidden Size    = %8d"   % config.hidden_size
  print "Num Steps      = %8d"   % config.num_steps
  print "Dropout        = %8.6f" % config.dropout
  print "LearningRate   = %8.6f" % config.lr
  print "--------------------"  
  print "Best Validation Loss = %f" % val
  print "--------------------"    

  return config

def generate_RNNLM(config):
  gen_config            = deepcopy(config)
  gen_config.train      = False
  gen_config.batch_size = 1
  gen_config.num_steps  = 1

  model_name = "RNNLM-%d"%gen_config.idx
  g = tf.get_default_graph()
  with g.name_scope(model_name) as scope:
    gen_model = RNNLM_Model(gen_config)
    gen_RNNLM_Model(gen_model)
  tf.reset_default_graph()

def RNNLM():
  config  = find_best_RNNLM()
  # generate_RNNLM(config)  

if __name__ == "__main__":
  RNNLM()
  
