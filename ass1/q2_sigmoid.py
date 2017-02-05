import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    
    ### YOUR CODE HERE
    x = 1/(1 + np.exp(-x))
    ### END YOUR CODE
    
    return x

def sigmoid_grad(f):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input f should be the sigmoid
    function value of your original input x. 
    """
    
    ### YOUR CODE HERE
    f = f*(1-f)
    ### END YOUR CODE
    
    return f

def test_sigmoid_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    assert np.amax(f - np.array([[0.73105858, 0.88079708], 
        [0.26894142, 0.11920292]])) <= 1e-6
    print g
    assert np.amax(g - np.array([[0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])) <= 1e-6
    print "You should verify these results!\n"

def test_sigmoid(): 
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    t = np.array(0)
    rs = np.array(0.5)
    rg = np.array(0.25)
    f = sigmoid(t)
    g = sigmoid_grad(f)
    print f
    assert np.amax(f - rs) <= 1e-6
    print g
    assert np.amax(g - rg) <= 1e-6
    t = np.array([[[ -1,  -2,  -3,  -4],
                   [  5,   6,   7,   8],
                   [ -9, -10, -11, -12]],
                  [[ 13,  14,  15,  16],
                   [-17, -18, -19, -20],
                   [ 21,  22,  23,  0]]])
    rs = np.array([[[  2.68941421e-01,   1.19202922e-01,   4.74258732e-02,
                       1.79862100e-02],
                    [  9.93307149e-01,   9.97527377e-01,   9.99088949e-01,
                       9.99664650e-01],
                    [  1.23394576e-04,   4.53978687e-05,   1.67014218e-05,
                       6.14417460e-06]],
                   [[  9.99997740e-01,   9.99999168e-01,   9.99999694e-01,
                      9.99999887e-01],
                    [  4.13993755e-08,   1.52299795e-08,   5.60279641e-09,
                       2.06115362e-09],
                    [  9.99999999e-01,   1.00000000e+00,   1.00000000e+00,
                       5.00000000e-01]]])
    rg = np.array([[[  1.96611933e-01,   1.04993585e-01,   4.51766597e-02,
                       1.76627062e-02],
                    [  6.64805667e-03,   2.46650929e-03,   9.10221180e-04,
                       3.35237671e-04],
                    [  1.23379350e-04,   4.53958077e-05,   1.67011429e-05,
                       6.14413685e-06]],
                   [[  2.26031919e-06,   8.31527336e-07,   3.05902133e-07,
                       1.12535149e-07],
                    [  4.13993738e-08,   1.52299793e-08,   5.60279637e-09,
                       2.06115361e-09],
                    [  7.58256124e-10,   2.78946866e-10,   1.02618802e-10,
                       2.50000000e-01]]])
    f = sigmoid(t)
    g = sigmoid_grad(f)
    print f
    assert np.amax(f - rs) <= 1e-6
    print g
    assert np.amax(g - rg) <= 1e-6
    ### END YOUR CODE

if __name__ == "__main__":
    test_sigmoid_basic();
    test_sigmoid()
