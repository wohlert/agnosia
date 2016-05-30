"""
preprocessing

Provides routines for preprocessing of data.
"""

import numpy as np
from scipy.signal import savgol_filter


def scale(input_matrix: np.array) -> np.array:
    """
    Scale the unit of measure from femtotesla to tesla.
    """
    return input_matrix * 1e12


def normalise(input_matrix: np.array, axis=0) -> np.array:
    """
    Normalises the data for input in certain classifiers.
    """
    from scipy.stats import zscore

    return zscore(input_matrix, axis=axis)


def min_max(x) -> np.array:
    """
    Uses minmax normalisation to scale input to the interval of 0-1.
    """
    return np.abs((np.min(x) - x) / (np.max(x) - np.min(x)))


def smooth(input_matrix: np.array, window: int=17, order: int=2) -> np.array:
    """
    Apply Savitzky-Golay filtering to smooth the signal.
    """
    assert window % 2 == 1, "Window size must be odd"
    return savgol_filter(input_matrix, window, order)


def remove_baseline(input_matrix: np.array, start: int) -> np.array:
    """
    Removes baseline noise from a signals.
    """
    start = int(start)
    baseline = np.mean(input_matrix[:, :, :start], axis=-1)
    return input_matrix - baseline[:, :, None]


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


def get_magnetometers(file: str) -> np.array:
    """
    Returns the input matrix with only the data
    from the magnetometers.

    Expected no. of magnetometers: 102
    """
    meters = np.load(file)

    find_magnetometers = np.vectorize(lambda x: bool(x.endswith("1")))
    (magnetometers,) = np.where(find_magnetometers(meters))

    return magnetometers


def get_gradiometers(file: str) -> np.array:
    """
    Returns the input matrix with only the data
    from the gradiometers.

    Expected no. of gradiometers: 204
    """
    meters = np.load(file)

    find_gradiometers = np.vectorize(lambda x: bool(not x.endswith("1")))
    (gradiometers,) = np.where(find_gradiometers(meters))

    return gradiometers


def keep_channels(input_matrix: np.array, type: str) -> np.array:
    """
    Remove channels from matrix that are
    not contained in channels.
    """
    channels = None

    if type == "gradiometers":
        channels = get_gradiometers("channel_names.npy")
    elif type == "magnetometers":
        channels = get_magnetometers("channel_names.npy")

    return input_matrix[:, channels, :]


def cut(input_matrix: np.array, start: int, end: int=None) -> np.array:
    """
    Removes samples before a given point,
    such as before stimuli.
    Can also trim from both sides.
    """
    _, _, samples = np.shape(input_matrix)

    start = int(start)

    if end != None:
        end = int(end)

    assert start < samples

    return input_matrix[:, :, start:end].copy()


def cut_m170(input_matrix: np.array, start: float, sfreq: int, window_size: float=5.0) -> np.array:
    """
    Cuts the samples around M170.
    window_size is the number of ms before and after n170.
    """
    window = window_size*0.01

    impulse = abs(start)
    prime = impulse + 0.170
    nmin = prime - window
    nmax = prime + window

    area = range(int(nmin*sfreq), int(nmax*sfreq))

    return input_matrix[:, :, area].copy()

