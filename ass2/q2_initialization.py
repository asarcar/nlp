import numpy as np
import tensorflow as tf

def xavier_weight_init():
  """
  Returns function that creates random tensor. 

  The specified function will take in a shape (tuple or 1-d array) and must
  return a random tensor of the specified shape and must be drawn from the
  Xavier initialization distribution.

  Hint: You might find tf.random_uniform useful.
  """
  def _xavier_initializer(shape, **kwargs):
    """Defines an initializer for the Xavier distribution.

    This function will be used as a variable scope initializer.

    https://www.tensorflow.org/versions/r0.7/how_tos/variable_scope/index.html#initializers-in-variable-scope

    Args:
      shape: Tuple or 1-d array that species dimensions of requested tensor.
    Returns:
      out: tf.Tensor of specified shape sampled from Xavier distribution.
    """
    ### YOUR CODE HERE
    eps = tf.sqrt(6.0/tf.cast(tf.reduce_sum(shape), tf.float32))
    out = tf.random_uniform(shape, minval=-eps, maxval=eps, name="XavierWeights")
    ### END YOUR CODE
    return out
  # Returns defined initializer function.
  return _xavier_initializer

def test_initialization_basic():
  """
  Some simple tests for the initialization.
  """
  print "Running basic tests..."
  xavier_initializer = xavier_weight_init()
  shape = (1,)
  xavier_mat = xavier_initializer(shape)
  assert xavier_mat.get_shape() == shape

  shape = (1, 2, 3)
  xavier_mat = xavier_initializer(shape)
  assert xavier_mat.get_shape() == shape
  print "Basic (non-exhaustive) Xavier initialization tests pass\n"

def test_initialization():
  """ 
  Use this space to test your Xavier initialization code by running:
      python q1_initialization.py 
  This function will not be called by the autograder, nor will
  your tests be graded.
  """
  print "Running your tests..."
  ### YOUR CODE HERE
  d1 = 4
  d2 = 20
  xavier_initializer = xavier_weight_init()
  t1    = tf.convert_to_tensor(np.zeros((d1,)))
  xm1 = xavier_initializer(tf.shape(t1))
  assert xm1.get_shape() == t1.get_shape()
  tm1    = tf.maximum(tf.reduce_max(xm1), tf.reduce_max(-xm1))
  t2     = tf.convert_to_tensor(np.zeros((d1,d2)))
  xm2 = xavier_initializer(tf.shape(t2))
  assert xm2.get_shape() == t2.get_shape()
  tm2    = tf.maximum(tf.reduce_max(xm2), tf.reduce_max(-xm2))
  with tf.Session() as s:
    m1, v1 = s.run([xm1, tm1])
    m2, v2 = s.run([xm2, tm2])
  val1 = np.sqrt(6.0/(d1*1.0))
  val2 = np.sqrt(6.0/((d1+d2)*1.0))
  print "Xavier Matrix: ", m1
  print "Max Val: ", v1, " should be less than: ", val1
  print "Xavier Matrix: ", m2
  print "Max Val: ", v2, " should be less than: ", val2
  assert v1 <= val1
  assert v2 <= val2
  print "Xavier initialization tests pass\n"
  ### END YOUR CODE  

if __name__ == "__main__":
    test_initialization_basic()
    test_initialization()
