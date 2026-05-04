import numpy as np
def relu(x):
    return np.maximum(0,x)
def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    W1 = np.array(W1)
    W2 = np.array(W2)
    W3 = np.array(W3)
    Ws = np.array(Ws)

    shortcut =  x @ Ws
    
    out = relu(x @ W1)
    out = relu(out @ W2)
    out = out @ W3

    return relu(out + shortcut)
