import numpy as np

def pool(X):
    """
    Creates features from pooling all the data
    in a specific interval across channels.
    """
    (m, n, o) = X.shape
    X_pool = X.reshape(m, n * o)
    X_pool -= X_pool.mean(axis = 0)
    X_pool = np.nan_to_num(X_pool / X_pool.std(axis = 0))
    return X_pool
