"""
preprocessing

Provides routines for preprocessing of data.
"""

import numpy as np
import scipy.signal as signal

def normalise(input_matrix: np.array) -> np.array:
    """
    Normalises the data for input in certain classifiers.
    Necessary for NN input.
    """
    from scipy.stats import zscore

    return zscore(input_matrix, axis=1)

def dropout_channels_tanh(input_matrix: np.array) -> np.array:
    """
    Finds channels to dropout based on the hyperbolic tangent
    along with the standard deviation of these tangents.

    ! input must be normalised.
    """
    tangents = np.tanh(input_matrix)
    cross_sample = np.std(tangents, axis=0)
    cross_trial = np.mean(cross_sample, axis=1)

    return cross_trial > 0.4

def dropout_channels_norm(input_matrix: np.array, threshold: float=0.05) -> np.array:
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

    return np.array(valid_channels)

def cut_samples(input_matrix: np.array, start: int, end: int=None) -> np.array:
    """
    Removes samples before a given point,
    such as before stimuli.
    Can also trim from both sides.
    """
    _, _, samples = input_matrix.shape

    assert start < samples
    assert end < samples

    return input_matrix[:, :, start:end].copy()

def cut_m170(input_matrix: np.array, tmin: float, sfreq: int, window_size: float=5.0) -> np.array:
    """
    Cuts the samples around M170.
    window_size is the number of ms before and after m170
    """
    window = window_size*0.01

    print(window)

    impulse = abs(tmin)
    prime = impulse + 0.170
    nmin = prime - window
    nmax = prime + window

    area = range(int(nmin*sfreq), int(nmax*sfreq))

    return input_matrix[:, :, area].copy()


def butter_lowpass_filter(data, nyquist: int, cutoff: float=5, order: int=6) -> np.array:
    """
    Creates a Butter windowed lowpass filter.
    """
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.lfilter(b, a, data)
