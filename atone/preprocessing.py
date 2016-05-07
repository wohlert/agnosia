"""
preprocessing

Provides routines for preprocessing of data.
"""

import numpy as np
from scipy.signal import savgol_filter, decimate


def scale(input_matrix: np.array) -> np.array:
    """
    Scale the unit of measure from femtotesla to tesla.
    """
    return input_matrix * 1e12


def normalise(input_matrix: np.array, axis=1) -> np.array:
    """
    Normalises the data for input in certain classifiers.
    """
    from scipy.stats import zscore

    return zscore(input_matrix, axis=axis)


def min_max(x, axis=0):
    """
    Uses minmax normalisation to scale input
    """
    return np.abs((np.min(x, axis=axis) - x) / (np.max(x, axis=axis) - np.min(x, axis=axis)))


def smooth(input_matrix: np.array, window: int=17, order: int=2) -> np.array:
    """
    Apply Savitzky-Golay filtering to smooth the signal.
    """
    assert window % 2 == 1, "Window size must be odd"
    return savgol_filter(input_matrix, window, order)


def dropout_channels_monte_carlo(input_matrix: np.array, output_labels: np.array) -> np.array:
    """
    Perform 10 fold shuffle split on the data and
    do cross validation to find channels with
    highest correlation to the output variable.
    """
    from sklearn.svm import SVC

    clf = SVC(C=1, kernel='linear')

    trials, channels, samples = np.shape(input_matrix)

    def monte_carlo_channel(channel):
        from sklearn.cross_validation import ShuffleSplit, cross_val_score
        from .features import pool

        cross_validation = ShuffleSplit(trials, n_iter=5, test_size=0.2)
        input_pooled = pool(input_matrix[:, [channel]])
        scores = cross_val_score(clf, input_pooled, output_labels, cv=cross_validation)

        return np.mean(scores)

    channel_list = np.arange(channels)
    accuracies = np.array([monte_carlo_channel(c) for c in channel_list])

    return accuracies


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

    trials, channels, _ = np.shape(input_matrix)
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


def cut(input_matrix: np.array, start: int, end: int=None) -> np.array:
    """
    Removes samples before a given point,
    such as before stimuli.
    Can also trim from both sides.
    """
    _, _, samples = np.shape(input_matrix)

    assert start < samples

    return input_matrix[:, :, start:end].copy()


def cut_m170(input_matrix: np.array, tmin: float, sfreq: int, window_size: float=5.0) -> np.array:
    """
    Cuts the samples around M170.
    window_size is the number of ms before and after n170.
    """
    window = window_size*0.01

    impulse = abs(tmin)
    prime = impulse + 0.170
    nmin = prime - window
    nmax = prime + window

    area = range(int(nmin*sfreq), int(nmax*sfreq))

    return input_matrix[:, :, area].copy()


def downsample(input_matrix: np.array, factor: int=2):
    """
    Downsamples the signal by a given factor.
    """
    return decimate(input_matrix, factor, ftype="fir")
