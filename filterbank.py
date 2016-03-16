from numpy.fft import rfft
import numpy as np
from scipy.stats.mstats import zscore

def fft(trial, limit):
    # Take the real fft of the trial
    transform = np.real(rfft(trial, axis=1))

    # Remove all higher frequencies
    transform = transform[:,1:limit]

    # Log10 of all values
    return np.log10(np.abs(transform))

def filter_bank(trial, nyquist):
    """
    Creates a filterbank with different bandpasses
    to separate the data. Then builds features
    from its eigenvalues.
    """
    transform = fft(trial, nyquist)

    # Different frequency bands
    bands = [(0,4), (4,8), (8,12), (12,30), (30,70), (70,nyquist-1)]

    # Apply z-score normalisation
    normalised = np.hstack([z_score(transform[:, band[0]:band[1]]) for band in bands])

    # Correlation coefficient matrix
    corr = np.corrcoef(normalised)

    # Eigenvalues
    eigenvalues = np.abs(np.linalg.eig(corr)[0])
    eigenvalues.sort()
    eigenvalues = eigenvalues[::-1]
    return corr
    #coeff = 
    #return np.concatenate((coeff, eigenvalues[::-1]))
