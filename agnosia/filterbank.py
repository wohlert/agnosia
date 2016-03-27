"""
filterbank

Provides the filterbank solution as described by Michael Hills.
"""

from numpy.fft import rfft
import numpy as np
from scipy.stats.mstats import zscore


def fft(trial, limit: int):
    """
    Transforms the trial using the discrete Fourier transform.
    Applies other functions to ease signal banding.
    """
    # Take the real fft of the trial
    transform = np.abs(rfft(trial, axis=1))

    # Remove all higher frequencies above limit
    transform = transform[:, 1:limit]

    # Log10 of all values to scale
    return np.log10((transform))

def upper_right_triangle(matrix) -> np.array:
    """
    Returns the upper right triangle of a m x m matrix.
    """
    matrix_length = len(matrix)
    m_half = int(matrix_length/2)
    limit = m_half
    out = []
    for i in range(m_half+1):
        for j in range(matrix_length):
            if j >= limit:
                out.append(matrix[i, j])
            if j == matrix_length-1:
                limit += 1
    return np.array(out)

def filter_bank(trial, limit: int=47):
    """
    Creates a filterbank with different bandpasses
    to separate the data. Then builds features
    from its eigenvalues.
    """
    transform = fft(trial, limit)

    # Different frequency bands
    #
    # delta = 0.1 - 3 Hz
    # theta = 4 - 7 Hz
    # alpha = 8 - 15 Hz / mu = 7.5 - 12 Hz
    # - SMR = 12.5 - 15.5 Hz
    # beta = 16 - 31 Hz
    # low-gamma = 32 - 64 Hz approx.
    # high-gamma = approx. 64 - 100 Hz
    bands = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 100)]

    # Apply z-score normalisation for each band
    normalised = np.hstack([zscore(transform[:, band[0]:band[1]]) for band in bands])

    # Correlation coefficient matrix
    corr = np.corrcoef(normalised)

    # Eigenvalues
    eigenvalues = np.abs(np.linalg.eig(corr)[0])
    eigenvalues.sort()
    eigenvalues = eigenvalues[::-1]
    coeff = upper_right_triangle(corr)
    return np.concatenate((coeff, eigenvalues))
