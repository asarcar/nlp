import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))  # dim: Dx/H
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))       # dim: 1/H
    ofs += H                                           
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy)) # dim: H/Dy
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))     # dim: 1/Dy

    ### YOUR CODE HERE: forward propagation
    # data -> W1 + b1 -> sigmoid -> W2 + b2 -> softmax
    N  = data.shape[0]
    z2 = data.dot(W1) + b1   # dim: (N/Dx)*(Dx/H) + (1/H@bcast) = N/H
    a2 = sigmoid(z2)         # dim: (N/H)
    z3 = a2.dot(W2) + b2     # dim: (N/H)*(H/Dy) + (1/Dy@bcast) = N/Dy
    y_hat = softmax(z3)      # dim: N/Dy
    l_pos = np.argmax(labels, axis=1)           # dim: N
    xent  = -np.log(y_hat[np.arange(N), l_pos]) # dim: N
    # x_entropy = avg_of_samples(sum_of_all_class(y_class_i*y_hat_class_i))
    cost = np.mean(xent, axis=0) # dim: 1 i.e. scalar
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    d3     = 1/N*(y_hat - labels)       # dim:                N/Dy
    gradb2 = np.sum(d3, axis=0)         # dim:sum(N/Dy)     = Dy
    gradW2 = a2.T.dot(d3)               # dim:(H/N)*(N/Dy)  = H/Dy
    grada2 = d3.dot(W2.T)               # dim:(N/Dy)*(Dy/H) = N/H
    d2     = grada2*sigmoid_grad(a2)    # dim:              = N/H
    gradb1 = np.sum(d2, axis=0)         # dim: sum(d2)        H
    gradW1 = data.T.dot(d2)             # dim: (Dx/N)*(N/H) = Dx/H
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
                           gradW2.flatten(), gradb2.flatten()))
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 1 # 20
    dimensions = [10, 5, 10] # [10, 5, 10] i.e. Dx/H/Dy
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
      labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
      dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
