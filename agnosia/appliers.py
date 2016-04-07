"""
appliers

Routines for applying preprocessing
"""

import numpy as np

def apply_dropout(input_matrix: np.array, area: np.array) -> np.array:
    return input_matrix[:, area, :].copy()

def apply_ica(input_matrix: np.array, components=None) -> np.array:
    from sklearn.decomposition import FastICA

    ica = FastICA(n_components=components)

    def ica_transform(trial: np.array) -> np.array:
        channels, samples = trial.shape
        trial = trial.reshape(samples, channels)
        transform = ica.fit_transform(trial)
        return transform

    transforms = np.dstack([ica_transform(trial) for trial in input_matrix])
    samples, channels, trials = transforms.shape
    return transforms.reshape(trials, channels, samples)

def apply_lowpass(x, filter, decimation: int=8) -> np.array:
    """
    Applies a low pass filter to data X given the input
    signals nyquist values and optional cutoff and
    decimation factor.
    """
    trials = len(x)
    total = []

    for i in range(trials):
        lowpass = filter
        lowpass = np.mean(lowpass, axis=0)
        lowpass = signal.decimate(lowpass, decimation, ftype="fir")
        total.append(lowpass)

    return np.vstack(total), len(lowpass)
