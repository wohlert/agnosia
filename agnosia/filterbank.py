"""
filterbank

Provides the filterbank solution as described by Michael Hills.
"""

from numpy.fft import rfft
import numpy as np
from scipy.stats.mstats import zscore


def fft(trial, lower_limit: int=None, upper_limit: int=None):
    """
    Transforms the trial using the discrete Fourier transform.
    Applies other functions to ease signal banding.
    """
    # Take the real fft of the trial
    transform = np.abs(rfft(trial, axis=1))

    # Remove all higher frequencies above limit
    transform = transform[:, lower_limit:upper_limit]

    # Log10 of all values to scale
    return np.log10((transform))

def upper_right_triangle(matrix) -> np.array:
    """
    Returns the upper right triangle of a m x m matrix.
    """
    m, n = matrix.shape
    acc = []
    for i in range(m):
        for j in range(i+1, n):
            acc.append(matrix[i, j])

    return np.array(acc)

def full_filter_bank(input_matrix, bands:list=None):
    """
    Creates a filterbank with different bandpasses
    to separate the data. Then builds features
    from its eigenvalues.
    """

    # Different frequency bands
    #
    # delta = 0.1 - 3 Hz
    # theta = 4 - 7 Hz
    # alpha = 8 - 15 Hz / mu = 7.5 - 12 Hz
    # - SMR = 12.5 - 15.5 Hz
    # beta = 16 - 31 Hz
    # low-gamma = 32 - 64 Hz approx.
    # high-gamma = approx. 64 - 100 Hz
    if bands == None:
        bands = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 100)]

    low = max(np.min(bands), 1)
    high = min(np.max(bands), 128)

    def normal_transform(trial):
        transform = fft(trial, lower_limit=low, upper_limit=high)
        normalised = np.hstack([zscore(transform[:, band[0]:band[1]]) for band in bands])
        # Correlation coefficient matrix
        corr = np.corrcoef(normalised)

        # Eigenvalues
        eigenvalues = np.abs(np.linalg.eig(corr)[0])
        eigenvalues.sort()
        eigenvalues = eigenvalues[::-1]
        coeff = upper_right_triangle(corr)
        return np.concatenate((coeff, eigenvalues))

    return np.vstack([normal_transform(trial) for trial in input_matrix])
