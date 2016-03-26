from numpy.fft import rfft
import numpy as np
from scipy.stats.mstats import zscore

def fft(trial, limit):
    # Take the real fft of the trial
    transform = np.abs(rfft(trial, axis=1))

    # Remove all higher frequencies above limit
    transform = transform[:,1:limit]

    # Log10 of all values to scale
    return np.log10((transform))

def upper_right_triangle(matrix):
    """
    Returns the upper right triangle of a m x m matrix.
    """
    m, _ = matrix.shape
    m_half = int(m/2)
    limit = m_half
    out = []
    for i in range(m_half+1):
        for j in range(m):
            if j >= limit: out.append(matrix[i, j])
            if j == m-1: limit += 1
    return np.array(out)

def filter_bank(trial, nyquist):
    """
    Creates a filterbank with different bandpasses
    to separate the data. Then builds features
    from its eigenvalues.
    """
    transform = fft(trial, 47)

    # Different frequency bands
    #
    # delta = 0.1 - 3 Hz
    # theta = 4 - 7 Hz
    # alpha = 8 - 15 Hz / mu = 7.5 - 12 Hz
    # - SMR = 12.5 - 15.5 Hz
    # beta = 16 - 31 Hz
    # low-gamma = 32 - 64 Hz approx.
    # high-gamma = approx. 64 - 100 Hz
    bands = [(0,4), (4,8), (8,16), (16,32), (32,64), (64,100)]

    # Apply z-score normalisation for each band
    normalised = np.hstack([z_score(transform[:, band[0]:band[1]]) for band in bands])

    # Correlation coefficient matrix
    corr = np.corrcoef(normalised)

    # Eigenvalues
    eigenvalues = np.abs(np.linalg.eig(corr)[0])
    eigenvalues.sort()
    eigenvalues = eigenvalues[::-1]
    coeff = upper_right_triangle(corr)
    return np.concatenate((coeff, eigenvalues))
