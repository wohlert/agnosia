"""
features

Provides functions for feature extraction and selection.
"""

import numpy as np
from sklearn.decomposition import FastICA


def pool(input_matrix: np.array) -> np.array:
    """
    Creates features from pooling all the data
    in a specific interval across channels.
    """
    trials, channels, samples = np.shape(input_matrix)
    pooled = np.reshape(input_matrix, (trials, channels*samples))
    pooled = np.nan_to_num(pooled / np.std(pooled, axis=0))
    return pooled


def _correlate(covariance: np.array, correlation: float=0.8, threshold: int=2) -> tuple:
    """
    Returns unique values with high and low
    cross correlation of a square matrix.
    """
    from functools import reduce

    horizontal, vertical = np.where(np.abs(np.corrcoef(covariance)) > correlation)
    no_diagonal = [(k, v) for k, v in zip(horizontal, vertical) if k != v]

    unique = set()
    for t in no_diagonal:
        if not tuple(reversed(t)) in unique:
            unique.add(t)

    unique = list(unique)
    reduction = reduce(lambda x,y: x + y, map(list, unique))
    (high_correlation, ) = np.where(np.bincount(reduction) > threshold)
    low_correlation = list(set(np.arange(len(covariance))) - set(high_correlation))

    return high_correlation, low_correlation


def ica(input_matrix: np.array, inverse: bool=False):
    """
    Performs ICA on an input in
    order to reduce dimensionality.
    """
    trials, channels, samples = np.shape(input_matrix)

    ica = FastICA(n_components=None)

    transform = ica.fit_transform(np.vstack(input_matrix))
    transform = np.reshape(transform, (-1, channels, samples))

    covariance = np.mean(transform, axis=0)

    high, low = _correlate(covariance)

    stacked = np.vstack(transform[:, low, :])

    if inverse:
        stacked = ica.inverse_transform(stacked)

    return np.reshape(inverse, (trials, -1, samples))

