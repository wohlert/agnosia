"""
preprocessing

Provides routines for preprocessing of data.
"""

import numpy as np
import scipy.signal as signal


def normalise(input_matrix):
    """
    Normalises the data for input in certain classifiers.
    Necessary for NN input.
    """
    from scipy.stats import zscore

    return zscore(input_matrix)

def dropout_channels(input_matrix, threshold: float=0.05):
    """
    Identifies channels with a low signal-to-noise ratio (snr)
    and returns a list of the channels with quality higher than
    `threshold` of all signals.
    """
    from scipy.special import ndtr

    trials, channels, _ = input_matrix.shape
    snr_channels = {}

    for trial in range(trials):
        for channel in range(channels):
            samples = input_matrix[trial, channel]
            mu = np.mean(samples)
            sigma = np.std(samples)

            # Signal to noise ratio
            snr = mu/sigma
            if channel not in snr_channels or snr_channels[channel] > snr:
                snr_channels[channel] = snr

    ratios = list(snr_channels.values())

    point_estimate = np.mean(ratios)
    standard_dev = np.std(ratios)

    def approved(x) -> bool:
        """
        Function to measure of value is above p = threshold
        """
        zscore = (point_estimate - x)/standard_dev
        if ndtr(zscore) >= threshold:
            return True
        return False

    valid_channels = [k for k, v in snr_channels.items() if approved(v)]

    return input_matrix[:, valid_channels, :]

def cut_samples(input_matrix, start: int, end: int=None):
    """
    Removes samples before a given point,
    such as before stimuli.
    Can also trim from both sides.
    """
    _, _, samples = input_matrix.shape

    assert start < samples
    assert end < samples

    return input_matrix[:, :, start:end].copy()

def butter_lowpass_filter(data, nyquist: int, cutoff: float, order: int=6) -> np.array:
    """
    Creates a Butter windowed lowpass filter.
    """
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.lfilter(b, a, data)

def apply_lowpass(x, nyquist: int, cutoff: float=5, decimation: int=8):
    """
    Applies a low pass filter to data X given the input
    signals nyquist values and optional cutoff and
    decimation factor.
    """
    trials = len(x)
    total = []

    for i in range(trials):
        lowpass = butter_lowpass_filter(x[i], nyquist, cutoff)
        lowpass = np.mean(lowpass, axis=0)
        lowpass = signal.decimate(lowpass, decimation, ftype="fir")
        total.append(lowpass)

    return np.vstack(total), len(lowpass)
