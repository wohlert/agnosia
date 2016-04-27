"""
appliers

Routines for applying preprocessing
"""

import numpy as np


def apply_dropout(input_matrix: np.array, channels: np.array) -> np.array:
    return input_matrix[:, channels, :].copy()


def apply_ica(input_matrix: np.array, components=None) -> np.array:
    from sklearn.decomposition import FastICA

    ica = FastICA(n_components=components)

    def ica_transform(trial: np.array) -> np.array:
        trial_channels, trial_samples = np.shape(trial)
        trial = np.reshape(trial, (trial_samples, trial_channels))
        transform = ica.fit_transform(trial)
        return transform

    transforms = np.dstack([ica_transform(trial) for trial in input_matrix])
    samples, channels, trials = np.shape(transforms)
    return np.reshape(transforms, (trials, channels, samples))
