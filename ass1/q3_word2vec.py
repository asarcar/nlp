import numpy as np
import random
import string

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    x = (x.T/np.linalg.norm(x.T,axis=0)).T
    ### END YOUR CODE
    
    return x

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ""

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               
    
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!                                                  
    
    ### YOUR CODE HERE
    # Data Types
    # predicted:     center word v_hat  => dim: d 
    # target: o/p word idx u_o          => dim: scalar intA
    # outputVectors: U                  => dim: N/d
    # dataset: provides sample context words (v_hat) and target word: u_o
    # Formulas
    # cost  = U*v_hat; y_hat = softmax(score); delta = y_hat - y
    # Grad_U = delta*v_hat'; Grad_v_hat = U'*delta
    score    = outputVectors.dot(predicted)
    y_hat    = softmax(score) # dim: (N/d)*d = N
    cost     = -np.log(y_hat[target])                # scalar
    assert not np.isnan(cost)
    delta    = y_hat; delta[target] -= 1             # dim: N
    gradPred = outputVectors.T.dot(delta)            # dim: (d/N)*N = d
    grad     = np.outer(delta,predicted)             # dim: N/d
    ### END YOUR CODE
    
    return cost, gradPred, grad

def dummy_data():
  N = 10
  d = 3
  tokens = [string.lowercase[i] for i in range(0,N)]
  # Interface to the dataset for negative sampling
  dataset = type('dummy', (), {})()
  def dummySampleTokenIdx():
    return random.randint(0,N-1)

  def getRandomContext(C):
    return tokens[random.randint(0,N-1)], \
      [tokens[random.randint(0,N-1)] for i in xrange(2*C)]
  dataset.sampleTokenIdx = dummySampleTokenIdx
  dataset.getRandomContext = getRandomContext

  random.seed(31415)
  np.random.seed(9265)
  dummy_vectors = normalizeRows(np.random.randn(2*N,d))
  dummy_tokens = dict([(string.lowercase[i], i) for i in range(N)])

  return dataset, dummy_vectors, dummy_tokens

def gradcheck_fn(cost_gradient_fn):
  dataset, wordVectors, _ = dummy_data()

  D = wordVectors.shape[0]
  d = wordVectors.shape[1]
  N = D/2
  inputVectors  = wordVectors[:N,:]
  outputVectors = wordVectors[N:,:]

  def serialize_params(vec, mat):
    params = np.empty(d*(1 + N))
    params[0:d] = vec
    params[d:]  = mat.flatten()
    return params
  
  def deserialize_params(params):
    # predicted Vector, output vector
    predV = params[0:d]
    opM   = np.reshape(params[d:], (N,d))
    return predV, opM

  def wrap_cg_fn():
    pred_idx, target_idx = random.randint(0,N-1), random.randint(0,N-1)
    params = serialize_params(inputVectors[pred_idx], outputVectors)
    def wcg_fn(params):
      predV, opV = deserialize_params(params)
      cost, gradV, gradM = cost_gradient_fn(predV, target_idx, opV, dataset)
      return cost, serialize_params(gradV, gradM)
    return params, wcg_fn
  
  numtests = 1
  for i in range(numtests):
    params, cg_fn = wrap_cg_fn()
    gradcheck_naive(cg_fn, params)

def test_softmaxCostAndGradient():
  print "Testing softmaxCostAndGradient..."
  print "==== Gradient check for softmaxCostAndGradient ===="
  gradcheck_fn(softmaxCostAndGradient)

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    """ Negative sampling cost function for word2vec models """
  
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE
    # Data Types
    # predicted:     center word v_hat  => dim: d 
    # target: o/p word idx u_o          => dim: scalar intA
    # outputVectors: U                  => dim: N/d
    # dataset: provides sample context words (v_hat) and target word: u_o
    # Formulas
    # cost  = -log(sigmoid(u_o'*v_hat)) - sum_k[log(sigmoid(-u_k'*v_hat))]
    # Grad_u= delta*v_hat'; Grad_v_hat = U'*delta
    #
    # Ensure all indices are unique and only one instance of target is in list
    lt_idxs    = [target] + [dataset.sampleTokenIdx() for i in range(K)]
    # indices   = np.array(lt_idxs)
    idxs       = np.array(lt_idxs)
    # idxs      = np.unique(indices)
    # K         = idxs.shape[0]-1
    l_pos_neg = [1] + [-1 for i in range(K)]
    l_one_hot = [1] + [ 0 for i in range(K)]
    pos_neg   = np.array(l_pos_neg)        # dim: 1+K
    one_hot   = np.array(l_one_hot)        # dim: 1+K
    U         = outputVectors[idxs]        # dim: (1+K)/d
    score     = U.dot(predicted)           # dim: (1+K)
    y_hat     = sigmoid(score)             # dim: (1+K)
    sig_score = one_hot+pos_neg*(y_hat-1)  # dim: (1+K)
    log_score = np.log(sig_score)          # dim: (1+K)
    cost      = -np.sum(log_score)         # dim: scalar
    assert not np.isnan(cost)
    delta     = y_hat - one_hot            # dim: (1+K)
    gradPred  = U.T.dot(delta)             # dim: d/(1+K)*(1+K) = d
    gradU     = np.outer(delta,predicted)  # dim: (1+K)/d
    grad   = np.zeros(outputVectors.shape) # dim: N/d
    for i, idx in enumerate(idxs):
      grad[idx] += gradU[i]                # dim: N/d
    ### END YOUR CODE

    return cost, gradPred, grad

def test_negSamplingCostAndGradient():
  print "Testing negSamplingCostAndGradient..."
  print "==== Gradient check for negSamplingCostAndGradient ===="
  gradcheck_fn(negSamplingCostAndGradient)

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                   
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    cost    = 0
    gradIn  = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    # Given: input: contextWord => predict output: contextWords
    ip_idx  = tokens[currentWord]
    predV   = inputVectors[ip_idx]
    for w in contextWords:
      tgt_idx = tokens[w]
      cost_tmp, gradIn_tmp, gradOut_tmp = \
        word2vecCostAndGradient(predV, tgt_idx, outputVectors, dataset)
      assert not np.isnan(cost_tmp)
      cost += cost_tmp
      gradIn[ip_idx] += gradIn_tmp
      gradOut += gradOut_tmp
    ### END YOUR CODE
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################
    
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    win_sz  = len(contextWords)
    predV   = np.zeros(inputVectors.shape[1])
    idxs    = np.empty(win_sz,dtype=np.int)
    # Given: input: contextWords => predict output: contextWord
    tgt_idx  = tokens[currentWord]
    for i, inp_word in enumerate(contextWords):
      inp_idx = tokens[inp_word]
      idxs[i] = inp_idx
      # prediction vector is the average embedding of contextWords
      predV   += inputVectors[inp_idx]/win_sz
    cost, gradIn_tmp, gradOut = \
        word2vecCostAndGradient(predV, tgt_idx, outputVectors, dataset)
    assert not np.isnan(cost)
    # update all the input vectors in the context window
    for idx in idxs:
      gradIn[idx] += gradIn_tmp/win_sz
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    # batchsize = 50
    batchsize = 1
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]/2
    inputVectors = wordVectors[:N,:]
    outputVectors = wordVectors[N:,:]
    for i in xrange(batchsize):
        # C1 = random.randint(1,C)
        C1 = 1
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N, :] += gin / batchsize / denom
        grad[N:, :] += gout / batchsize / denom
        
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset, dummy_vectors, dummy_tokens = dummy_data()
    N = dummy_vectors.shape[0]/2
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:N,:], dummy_vectors[N:,:], dataset)
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:N,:], dummy_vectors[N:,:], dataset, negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:N,:], dummy_vectors[N:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:N,:], dummy_vectors[N:,:], dataset, negSamplingCostAndGradient)

if __name__ == "__main__":
    test_normalize_rows()
    test_softmaxCostAndGradient()
    test_negSamplingCostAndGradient()
    test_word2vec()
