"""
frequency

Provides the frequency based routines.
"""

import numpy as np
from numpy.fft import rfft
import scipy.signal as signal


def bandpass(input_matrix: np.array, fs: float, lowcut: float=0, highcut: float=None, order: int=5) -> np.array:
    """
    Applies a Butter bandpass filter to data X given the
    input signals nyquist values and optional cutoff and
    decimation factor.
    """
    nyq = 0.5 * fs

    if not highcut:
        highcut = nyq

    low = lowcut / nyq
    high = highcut / nyq
    numerator, denominator = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(numerator, denominator, input_matrix)


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
    return np.log10(transform)


def filter_bank(input_matrix, bands: list=None) -> np.array:
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
    if not bands:
        bands = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 100)]

    low = max(np.min(bands), 1)
    high = min(np.max(bands), 128)

    def single_bank(trial):
        transform = fft(trial, lower_limit=low, upper_limit=high)
        bank = np.hstack([transform[:, band[0]:band[1]] for band in bands])
        return bank

    transforms = np.dstack([single_bank(trial) for trial in input_matrix])
    return transforms


def dwt():
    pass


def wavelet():
    pass
