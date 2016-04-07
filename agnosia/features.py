"""
features

Provides functions for feature extraction and selection.
"""

import numpy as np

def pool(input_matrix: np.array) -> np.array:
    """
    Creates features from pooling all the data
    in a specific interval across channels.
    """
    trials, channels, samples = input_matrix.shape
    pooled = input_matrix.reshape(trials, channels*samples)
    pooled -= pooled.mean(axis=0)
    pooled = np.nan_to_num(pooled / pooled.std(axis=0))
    return pooled
