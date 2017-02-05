import getopt
import itertools
import math
import matplotlib
matplotlib.use('Agg') # Prevent connect to X11
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import tensorflow as tf
import time
import tree as tr

from collections import Counter
from collections import OrderedDict
from distutils.util import strtobool
from utils import Vocab
from sklearn.manifold import TSNE
from tensorflow.contrib.tensorboard.plugins import projector

RESET_AFTER = 50
result_dir = './results/'

class Config(object):
  """Holds model hyperparams and data information.
     Model objects are passed a Config() object at instantiation.
  """
  embed_size = 35
  label_size = 2
  early_stopping = 2
  anneal_threshold = 0.99
  anneal_by = 1.5
  max_epochs = 30
  lr = 0.01
  l2 = 0.02
  model_name = 'rnn5_hlhr_embed=%d_l2=%f_lr=%f.weights'%(embed_size, l2, lr)
  weight_dir = result_dir + 'weights/'

class RNN_Model():

  def load_data(self, debug=False):
    """Loads train/dev/test data and builds vocabulary."""
    n_train = 700; n_dev = 100; n_test = 200
    if debug:
      n_train = 20; n_dev = 5; n_test = 10
    self.train_data, self.dev_data, self.test_data = \
        tr.simplified_data(n_train, n_dev, n_test)

    # build vocab from training data
    self.vocab = Vocab()
    train_sents = [t.get_words() for t in self.train_data]
    self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))

  def inference(self, tree, predict_only_root=False):
    """For a given tree build the RNN models computation graph up to where it
        may be used for inference.
    Args:
        tree: a Tree object on which to build the computation graph for the RNN
    Returns:
        softmax_linear: Output tensor with the computed logits.
    """
    node_tensors = self.add_model(tree.root)
    if predict_only_root:
        node_tensors = node_tensors[tree.root]
    else:
        node_tensors = [tensor for node, tensor in node_tensors.iteritems() if node.label!=2]
        node_tensors = tf.concat(0, node_tensors)
    return self.add_projections(node_tensors)

  def add_model_vars(self):
    '''
    You model contains the following parameters:
    COMPOSITION: h = max([hl, hr]W1 + b1, 0)
        embedding:  tensor(vocab_size, embed_size)
        W1:         tensor(2* embed_size, embed_size)
        b1:         tensor(1, embed_size)
    PROJECTION: a = hU + bs
        U:          tensor(embed_size, output_size)
        bs:         tensor(1, output_size)
    Hint: Add the tensorflow variables to the graph here and *reuse* them while building
            the compution graphs for composition and projection for each tree
    Hint: Use a variable_scope "Composition" for the composition layer, and
          "Projection") for the linear transformations preceding the softmax.
    '''
    V = len(self.vocab)
    d = self.config.embed_size
    o = self.config.label_size
    # Trust default Xavier initializer tf.uniform_unit_scaling_initializer
    with tf.variable_scope('Composition'):
      embedding = tf.get_variable("Embedding", [V, d])
      W1        = tf.get_variable("W1", [2*d, d])
      b1        = tf.get_variable("b1", [1, d])
    with tf.variable_scope('Projection'):
      U         = tf.get_variable("U", [d, o])
      bs        = tf.get_variable("bs", [1, o])
    with tf.variable_scope('Training', initializer=tf.constant(0)):
      step      = tf.get_variable("global_step", trainable=False, dtype=tf.int32)
      
  def add_model(self, node):
    """Recursively build the model to compute the phrase embeddings in the tree

    Hint: Refer to tree.py and vocab.py before you start. Refer to
          the model's vocab with self.vocab
    Hint: Reuse the "Composition" variable_scope here
    Hint: Store a node's vector representation in node.tensor so it can be
          used by it's parent
    Hint: If node is a leaf node, it's vector representation is just that of the
          word vector (see tf.gather()).
    Args:
        node: a Node object
    Returns:
        node_tensors: Dict: key = Node, value = tensor(1, embed_size)
    """
    with tf.variable_scope('Composition', reuse=True):
      ### YOUR CODE HERE
      embedding = tf.get_variable("Embedding")
      W1        = tf.get_variable("W1")
      b1        = tf.get_variable("b1")          
      ### END YOUR CODE

    d = self.config.embed_size
    # Imperative to add tensors in postorder traversal so that
    # it matches exact position of the list of labels of the tree
    # Otherwise: loss calculations would be incorrect
    node_tensors = OrderedDict()
    if node.isLeaf:
      ### YOUR CODE HERE
      # Formula:
      # COMPOSITION: Embed Vector
      word_idx = self.vocab.encode(node.word)
      curr_node_tensor = tf.reshape(embedding[word_idx], [1, d])
      ### END YOUR CODE
    else:
      # Formula:
      # COMPOSITION: h = max([hl, hr]W1 + b1, 0)
      node_tensors.update(self.add_model(node.left))
      node_tensors.update(self.add_model(node.right))
      ### YOUR CODE HERE
      hl = node_tensors[node.left]
      hr = node_tensors[node.right]
      h  = tf.concat(1, [hl, hr], name="h")
      z  = tf.matmul(h, W1) + b1
      curr_node_tensor = tf.nn.relu(z, "ReLU")
      ### END YOUR CODE
    node_tensors[node] = curr_node_tensor
    return node_tensors

  def add_projections(self, node_tensors):
    """Add projections to the composition vectors to compute the raw sentiment scores

    Hint: Reuse the "Projection" variable_scope here
    Args:
        node_tensors: tensor(?, embed_size)
    Returns:
        output: tensor(?, label_size)
    """
    ### YOUR CODE HERE
    # PROJECTION: a = hU + bs
    with tf.variable_scope('Projection', reuse=True):
      ### YOUR CODE HERE
      # U [d, o], bs [1, o]
      U         = tf.get_variable("U")
      bs        = tf.get_variable("bs")
      ### END YOUR CODE
    logits    = tf.matmul(node_tensors, U) + bs
    ### END YOUR CODE
    return logits

  def loss(self, logits, labels):
    """Adds loss ops to the computational graph.

    Hint: Use sparse_softmax_cross_entropy_with_logits
    Hint: Remember to add l2_loss (see tf.nn.l2_loss)
    Args:
        logits: tensor(num_nodes, output_size)
        labels: python list, len = num_nodes
    Returns:
        loss: tensor 0-D
    """
    # YOUR CODE HERE
    # Dim: [Num_Nodes]
    xent_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name="SparseSoftmax")
    xent_loss   = tf.reduce_sum(xent_losses)
    with tf.variable_scope('Composition', reuse=True):
      W1        = tf.get_variable("W1")
    with tf.variable_scope('Projection', reuse=True):
      U         = tf.get_variable("U")
    l2_loss = tf.nn.l2_loss(W1, name="W1Loss")
    l2_loss += tf.nn.l2_loss(U, name="ULoss")
    loss = self.config.l2*l2_loss + xent_loss
    # END YOUR CODE
    return loss

  def training(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.GradientDescentOptimizer for this model.
            Calling optimizer.minimize() will return a train_op object.

    Args:
        loss: tensor 0-D
    Returns:
        train_op: tensorflow op for training.
    """
    # YOUR CODE HERE
    with tf.variable_scope('Training', reuse=True):
      step = tf.get_variable("global_step", dtype=tf.int32)
    optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
    train_op = optimizer.minimize(
      loss, global_step=step, name="MinimizeGradientDescent")
    # END YOUR CODE
    return train_op

  def predictions(self, y):
    """Returns predictions from sparse scores

    Args:
        y: tensor(?, label_size)
    Returns:
        predictions: tensor(?,1)
    """
    predictions = None
    # YOUR CODE HERE
    predictions = tf.argmax(y, axis=1, name="Predictions")
    # END YOUR CODE
    return predictions

  def __init__(self, config, debug=False):
    self.config = config
    self.load_data(debug)

  def predict(self, trees, weights_path, get_loss = False):
    """Make predictions from the provided model."""
    results = []
    losses = []
    for i in xrange(int(math.ceil(len(trees)/float(RESET_AFTER)))):
        with tf.Graph().as_default(), tf.Session() as sess:
            self.add_model_vars()
            saver = tf.train.Saver()
            saver.restore(sess, weights_path)
            for tree in trees[i*RESET_AFTER: (i+1)*RESET_AFTER]:
                logits = self.inference(tree, True)
                predictions = self.predictions(logits)
                root_prediction = sess.run(predictions)[0]
                if get_loss:
                    root_label = tree.root.label
                    loss = sess.run(self.loss(logits, [root_label]))
                    losses.append(loss)
                results.append(root_prediction)
    return results, losses

  def initialize_model(self, sess, new_model = False):
    # If prior incarnation of run available just use it
    wt_dir = self.config.weight_dir
    model_name = '%s%s.temp'%(wt_dir,self.config.model_name)
    dfile  = model_name + '.data-00000-of-00001'
    ifile  = model_name + '.index'
    mfile  = model_name + '.meta'
    cfile  = wt_dir   + 'checkpoint'
    checkpoint_exists = os.path.exists(wt_dir) and \
                        os.path.isfile(dfile) and \
                        os.path.isfile(ifile) and \
                        os.path.isfile(mfile) and \
                        os.path.isfile(cfile)
    assert new_model or checkpoint_exists
    self.add_model_vars()
    if new_model:
      init = tf.global_variables_initializer()
      sess.run(init)

    if checkpoint_exists:
      saver = tf.train.Saver()
      saver.restore(sess, model_name)

  def run_epoch(self, new_model = False, verbose=True):
    step = 0
    loss_history = []
    while step < len(self.train_data):
        with tf.Graph().as_default(), tf.Session() as sess:
            self.initialize_model(sess, new_model)
            for _ in xrange(RESET_AFTER):
                if step>=len(self.train_data):
                    break
                tree = self.train_data[step]
                logits = self.inference(tree)
                labels = [l for l in tree.labels if l!=2]
                loss = self.loss(logits, labels)
                train_op = self.training(loss)
                loss, _ = sess.run([loss, train_op])
                loss_history.append(loss)
                if verbose:
                    sys.stdout.write('\r{} / {} :    loss = {}'.format(
                        step, len(self.train_data), np.mean(loss_history)))
                    sys.stdout.flush()
                step+=1
            saver = tf.train.Saver()
            if not os.path.exists(self.config.weight_dir):
              os.makedirs(self.config.weight_dir)
            saver.save(sess, '%s%s.temp'%(self.config.weight_dir,self.config.model_name))
    train_preds, _ = self.predict(self.train_data, '%s%s.temp'%(self.config.weight_dir,self.config.model_name))
    val_preds, val_losses = self.predict(self.dev_data, '%s%s.temp'%(self.config.weight_dir,self.config.model_name), get_loss=True)
    train_labels = [t.root.label for t in self.train_data]
    val_labels = [t.root.label for t in self.dev_data]
    train_acc = np.equal(train_preds, train_labels).mean()
    val_acc = np.equal(val_preds, val_labels).mean()

    print
    print 'Training acc (only root node): {}'.format(train_acc)
    print 'Validation acc (only root node): {}'.format(val_acc)
    print self.make_conf(train_labels, train_preds)
    print self.make_conf(val_labels, val_preds)
    return train_acc, val_acc, loss_history, np.mean(val_losses)

  def train(self, verbose=True):
    complete_loss_history = []
    train_acc_history = []
    val_acc_history = []
    prev_epoch_loss = float('inf')
    best_val_loss = float('inf')
    best_val_epoch = 0
    stopped = -1
    for epoch in xrange(self.config.max_epochs):
        print 'epoch %d'%epoch
        if epoch==0:
            train_acc, val_acc, loss_history, val_loss = self.run_epoch(new_model=True)
        else:
            train_acc, val_acc, loss_history, val_loss = self.run_epoch()
        complete_loss_history.extend(loss_history)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        #lr annealing
        epoch_loss = np.mean(loss_history)
        if epoch_loss>prev_epoch_loss*self.config.anneal_threshold:
            self.config.lr/=self.config.anneal_by
            print 'annealed lr to %f'%self.config.lr
        prev_epoch_loss = epoch_loss

        #save if model has improved on val
        if val_loss < best_val_loss:
          fname_tmp = '%s%s.temp'%(self.config.weight_dir,self.config.model_name)
          fname     = '%s%s'%(self.config.weight_dir,self.config.model_name)
          shutil.copyfile(fname_tmp + ".index", fname + ".index")
          shutil.copyfile(fname_tmp + ".meta", fname + ".meta")
          shutil.copyfile(fname_tmp + ".data-00000-of-00001", fname + ".data-00000-of-00001")
          best_val_loss = val_loss
          best_val_epoch = epoch

        # if model has not imprvoved for a while stop
        if epoch - best_val_epoch > self.config.early_stopping:
            stopped = epoch
            #break
    if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()

    print '\n\nstopped at %d\n'%stopped
    return {
        'loss_history': complete_loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
        }

  def make_conf(self, labels, predictions):
    confmat = np.zeros([2, 2])
    for l,p in itertools.izip(labels, predictions):
        confmat[l, p] += 1
    return confmat

  def embeds(self, num, wt_path):
    """Extract topmost 'num' embed from the provided model.
    Extract the corresponding labels and corresponding indices.
    """
    V = len(self.vocab)
    d = self.config.embed_size
    v = self.vocab

    assert num > 0
    # Skip first 100 words as many are useless/uninteresting
    # e.g. '.', 'a', ',', 'and', 'the', 'is', 'of', 'it', "'s"
    num_useless = 100
    word_count = Counter(v.word_freq).most_common(num+num_useless)
    word_count = word_count[num_useless:]
    words      = [w for w,count in word_count]
    words = [w.decode('unicode-escape') for w in words]
    idxs       = [v.encode(w) for w in words]

    with tf.Graph().as_default(), tf.Session() as sess:
      with tf.variable_scope('Composition'):
        embeddings = tf.get_variable("Embedding", [V, d])
      saver = tf.train.Saver()
      saver.restore(sess, wt_path)
      # Extract the embeddings corresponding to idxs
      tfembeds = tf.nn.embedding_lookup(embeddings, idxs)
      embeds = sess.run(tfembeds)

    self.embed_save(embeds, words)

  def annotate_data_save(self):
    """Save annotated data in TSV (Tab Separated Value) format"""
    # Save annotated data to TSV Metadata file
    v       = self.vocab
    idxs    = range(len(v))
    words   = [v.decode(i) for i in idxs]

    metadata_fname = '%s%s-metadata.tsv'%(self.config.weight_dir,self.config.model_name)
    with open(metadata_fname, "w") as f:
      f.write("Index\tLabel\n")
      index = 0
      for word in words:
        f.write("%d\t%s\n"%(index,word))
        index += 1

    return metadata_fname
    
  def embeds_all(self):
    """Fulfill requirements to visualize embeddings in TensorBoard
    1. Save annotated data in TSV (Tab Separated Value) format
    2. Restore Embeddings Tensor
    3. Format TensorBoard (TB) Data: Link embedding tensor to metadatafile
    4. Save TB projector with visualization of embedding to config file.
    """

    # 1. Save annotated data to TSV Metadata file
    metadata_fname = self.annotate_data_save()

    # 2. Restore Embeddings Tensor
    wt_path = '%s%s'%(self.config.weight_dir,self.config.model_name)
    with tf.Graph().as_default(), tf.Session() as sess:
      V = len(self.vocab)
      d = self.config.embed_size
      with tf.variable_scope('Composition'):
        embeddings = tf.get_variable("Embedding", [V, d])

      # 3. Format TensorBoard (TB) Data: Link embedding tensor to metadatafile
      saver = tf.train.Saver()
      saver.restore(sess, wt_path)
      # dir should be same where we save session & stored checkpoint.
      summary_writer = tf.summary.FileWriter(self.config.weight_dir)
      
      # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
      config = projector.ProjectorConfig()
      # Add embedding to TB - multiple embeddings can be added
      emb_tb = config.embeddings.add()
      emb_tb.tensor_name = embeddings.name
      # Link this TB tensor to its metadata file (e.g. labels).
      emb_tb.metadata_path = metadata_fname

      # 4. Save TB projector with visualization of embedding to config file.
      projector.visualize_embeddings(summary_writer, config)

  def embed_save(self, embeds, words):
    tsne = TSNE(n_components=2, init='pca')
    emb2d = tsne.fit_transform(embeds)
  
    plt.figure(figsize=(20,20)) # in inches
    for i, word in enumerate(words):
      x, y = emb2d[i,:]
      plt.scatter(x, y)
      plt.annotate(word, xy=(x,y), xytext=(5,2), textcoords='offset points',
                   ha='right', va='bottom')
    fig_fname = '%s%s-%d-embeds.png'%(self.config.weight_dir, self.config.model_name,len(words))
    plt.savefig(fig_fname)
    
  def loss_history_save(self, stats):
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    fig_fname = '%s%s-loss_history.png'%(self.config.weight_dir,self.config.model_name)
    plt.savefig(fig_fname)

def test_acc(model):
  print 'Test'
  print '=-=-='
  model_name = '%s%s'%(model.config.weight_dir,model.config.model_name)
  predictions, _ = model.predict(model.test_data, model_name)
  labels = [t.root.label for t in model.test_data]
  test_acc = np.equal(predictions, labels).mean()
  print 'Test acc: {}'.format(test_acc)

def train_RNN(debug):
  config = Config()
  if debug:
    config.max_epochs = 1
  model = RNN_Model(config, debug)
  start_time = time.time()
  stats = model.train(verbose=True)
  print 'Training time: {}'.format(time.time() - start_time)
  model.loss_history_save(stats)
  return model

def train_n_test_RNN(debug=False):
  """Train, Display Loss History, and Test RNN model implementation.  
  You can use this function to test your implementation of the Named Entity
  Recognition network. When debugging, set max_epochs in the Config object to 1
  so you can rapidly iterate.
  """
  model = train_RNN(debug)
  test_acc(model)

def board_embed_RNN():
  """Extract Embedding, Save Annotation, and save configuration
  for Tensorboard to visualize the embedding
  """  
  mconfig        = Config()
  model          = RNN_Model(mconfig)
  model.embeds_all()
  
def figure_embed_RNN(num):
  """Extract and using TSNE Save Visualization of 
  Top Few Final Embedding of RNN model.
  """  
  config = Config()
  model  = RNN_Model(config)
  wt_path = '%s%s'%(model.config.weight_dir,model.config.model_name)
  # extract the top 'Num' embeddings with its 'text' labels
  model.embeds(num, wt_path)

def main(argv):
  usage_str = 'rnn.py -h|--help -t|--train=<debug_flag> -b|--board_embed -f|--figure_embed=<N>'
  try:
    opts, args = getopt.getopt(argv,"ht:bf:",["help","train=","board_embed","figure_embed="])
  except getopt.GetoptError:
    print usage_str
    sys.exit(2)
  
  for opt, arg in opts:
    if opt in ("-h", "--help"):
      print usage_str
      sys.exit()
    if opt in ("-t", "--train"):
      print "Training Sentiment via Recursive Neural Network Language..."
      debug=bool(strtobool(arg))
      train_n_test_RNN(debug)
    if opt in ("-b", "--board_embed"):
      print "TensorBoard Configuration to visualize Embed Matrix..."
      board_embed_RNN()
    if opt in ("-f", "--figure_embed"):
      num = int(arg)
      print "Visualize Top %d Results of Embed Matrix..."%num 
      figure_embed_RNN(num)

if __name__ == "__main__":
  main(sys.argv[1:])
