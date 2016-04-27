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
    trials, channels, samples = np.shape(input_matrix)
    pooled = np.reshape(input_matrix, (trials, channels*samples))
    pooled = np.nan_to_num(pooled / np.std(pooled, axis=0))
    return pooled


def components(input_matrix: np.array) -> np.array:
    """
    Identifies the three prime components for facial recognition.

    * N170  Face perception
    * N250  Implicit facial recognition
    * P600f Explicit facial recognition
    """
    return input_matrix.copy()
