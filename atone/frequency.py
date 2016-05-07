"""
frequency

Provides the frequency based routines.
"""

import numpy as np
from numpy.fft import rfft
import scipy.signal as signal
import pywt as wave

from .constants import *


def bandpass(input_matrix: np.array, fs: float, lowcut: float=0, highcut: float=None, order: int=5) -> np.array:
    """
    Applies a Butter bandpass filter to data X given the
    input signals along with an upper and lower bandstop
    frequency.
    """
    nyq = 0.5 * fs

    if not highcut:
        highcut = nyq

    low = lowcut / nyq
    high = highcut / nyq
    numerator, denominator = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(numerator, denominator, input_matrix)


def fft(input_matrix: np.array, lower_limit: int=None, upper_limit: int=None):
    """
    Transforms the trial using the real-valued fast
    Fourier transform within an optional band.
    """
    # Take the real fft of the trial
    transform = np.abs(rfft(input_matrix, axis=-1))

    # Apply bandstop
    transform = transform[:, :, lower_limit:upper_limit]

    # Log10 of all values to scale
    return np.log10(transform)


class Filterbank:
    def __init__(self, bands: list=None):
        if not bands:
            self.__bands = [DELTA, THETA, ALPHA, BETA, LOW_GAMMA, HIGH_GAMMA]

    def bands(self):
        return self.__bands

    def apply(self, input_matrix: np.array) -> np.array:
        return filter_bank(input_matrix, self.__bands)


def filter_bank(input_matrix: np.array, bands: list=None) -> np.array:
    """
    Creates a filterbank with different bandpasses
    to separate the data. Then builds features
    from its eigenvalues.
    """
    if not bands:
        bands = [DELTA, THETA, ALPHA, BETA, LOW_GAMMA, HIGH_GAMMA]

    low = max(np.min(bands), 1)
    high = min(np.max(bands), 128)

    def single_bank(trial):
        transform = fft(trial, lower_limit=low, upper_limit=high)
        bank = np.hstack([transform[:, band[0]:band[1]] for band in bands])
        return bank

    transforms = np.dstack([single_bank(trial) for trial in input_matrix])
    channels, _, trials = np.shape(transforms)
    return np.reshape(transforms, (trials, channels, -1))


def dwt(input_matrix: np.array, order: int=3) -> np.array:
    """
    Applies a discrete wavelet transform to the data.
    """
    trials, channels, _ = np.shape(input_matrix)
    wavelet = wave.Wavelet("db{}".format(order))

    transform = np.array(wave.dwt(input_matrix, wavelet))
    return np.reshape(transform, (trials, channels, -1))


def dwt_bank(input_matrix: np.array, level: int, wave_type: str) -> tuple:
    """
    Applies a filtering of `level` number of levels
    to find approximation and details for a signal
    using wavelet transformation.
    """
    wavelet = wave.Wavelet(wave_type)
    approx, *details = wave.wavedec(input_matrix, wavelet, level=level)

    return approx, details


def dwt_spectrum(input_matrix: np.array, level: int=4, wave_type: str="db2") -> np.array:
    """
    Retrieves the full wavelet decomposition as a spectrum.
    """
    approx, details = dwt_bank(input_matrix, level, wave_type)
    spectrum = np.dstack([approx, np.dstack(details)])

    return spectrum
