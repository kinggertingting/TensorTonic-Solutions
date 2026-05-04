import numpy as np

def relu(x):
    return np.maximum(0, x)

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    # YOUR CODE HERE

    W1 = np.array(W1)
    W2 = np.array(W2)
    
    temp = x.copy()

    out = x @ W1.T
    out = relu(out)

    out = out @ W2.T
    out = relu(out + temp)

    return out
