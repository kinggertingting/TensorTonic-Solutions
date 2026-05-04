import numpy as np

def relu(x):
    return np.maximum(0, x)

#Identity block without same dimension
def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """
    W1 = np.array(W1)
    W2 = np.array(W2)
    Ws = np.array(Ws)
    
    shortcut = x @ Ws

    out = relu(x @ W1)

    out = out @ W2

    out = relu(out + shortcut)

    return out

    
    
