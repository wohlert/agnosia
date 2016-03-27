import numpy as np
import scipy.signal as signal

def normalise(X):
    from scipy.stats import zscore
    """
    Normalises the data for input in certain classifiers.
    Necessary for NN input.
    """
    return zscore(X)

def dropout_channels(X, threshold: float = 0.05):
    from scipy.special import ndtr
    """
    Identifies channels with a low signal-to-noise ratio (snr)
    and returns a list of the channels with quality higher than
    `threshold` of all signals.
    """
    m, n, o = X.shape
    channels = {}

    for trial in range(m):
        for channel in range(n):
            samples = X[trial, channel]
            mu = np.mean(samples)
            sigma = np.std(samples)

            # Signal to noise ratio
            snr = mu/sigma
            if channel not in channels or channels[channel] > snr:
                channels[channel] = snr

    ratios = list(channels.values())

    point_estimate = np.mean(ratios)
    standard_dev = np.std(ratios)

    # Function to measure of value is above p = threshold
    def approved(x) -> bool:
        zscore = (point_estimate - x)/standard_dev
        if ndtr(zscore) >= threshold: return True
        return False

    valid_channels = [k for k, v in channels.items() if approved(v)]

    return X[:, valid_channels, :]

def cut_samples(X, tmin: int, after: bool = False):
    """
    Removes samples before a given point,
    such as before stimuli.
    Can also trim from both sides.
    """
    m, n, o = X.shape

    discarded_left = tmin
    discarded_right = None

    if after: discarded_right = o - discarded_left

    return X[:, :, discarded_left:discarded_right].copy()

def butter_lowpass_filter(data, nyquist: int, cutoff: float, order: int = 6) -> np.array:
    """
    Creates a Butter windowed lowpass filter.
    """
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.lfilter(b, a, data)

def apply_lowpass(X, nyquist: int, cutoff: float = 5, decimation: int = 8):
    """
    Applies a low pass filter to data X given the input
    signals nyquist values and optional cutoff and
    decimation factor.
    """
    m, n, o = X.shape
    total = []
    for i in range(m):
        lowpass = butter_lowpass_filter(X[i], nyquist, cutoff)
        lowpass = np.mean(lowpass, axis = 0)
        lowpass = signal.decimate(lowpass, decimation, ftype = "fir")
        total.append(lowpass)

    return np.vstack(total), len(lowpass)
