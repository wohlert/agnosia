"""
io

Provides input and output operations for loading
MEG data and creating submissions.
"""

import numpy as np
from scipy.io import loadmat


def get_files(folder: str):
    """
    Retrieves a list of string of the filenames
    of .mat files in a folder.
    """
    import os

    train_path = "{}/train/".format(folder)
    train_files = ["{}{}".format(train_path, f) \
        for f in os.listdir(train_path) \
        if f.endswith(".mat")]

    test_path = "{}/test/".format(folder)
    test_files = ["{}{}".format(test_path, f) \
        for f in os.listdir(test_path) \
        if f.endswith(".mat")]

    return train_files, test_files

def load_subjects(folder: str, no_of_subjects: int=0):
    """
    Loads a number of subjects and splits the data accordingly.
    """
    from sklearn.cross_validation import train_test_split

    train_files, test_files = get_files(folder)

    # No subjects chosen, do true split
    if not no_of_subjects:
        train_subjects = [loadmat(p) for p in train_files]
        test_subjects = [loadmat(p) for p in test_files]

        x_train = np.vstack([subject['X'] for subject in train_subjects])
        y_train = np.vstack([subject['y'] for subject in train_subjects])

        x_test = np.vstack([subject['X'] for subject in test_subjects])
        y_test = np.vstack([subject['Id'] for subject in test_subjects])

    # Do a split for cross validation
    else:
        train_subjects = [loadmat(p) for p in train_files[:no_of_subjects]]
        x_full = np.vstack([subject['X'] for subject in train_subjects])
        y_full = np.vstack([subject['y'] for subject in train_subjects])

        x_train, x_test, y_train, y_test = train_test_split(x_full, y_full)

    return x_train, x_test, y_train, y_test

def load_meta(folder: str):
    """
    Loads the meta data for a .mat files in a folder.
    """
    train_files, _ = get_files(folder)
    subject = loadmat(train_files[0])

    sfreq = int(subject['sfreq'])
    tmin = int(subject['tmin'])
    tmax = int(subject['tmax'])

    return sfreq, tmin, tmax

def create_submission(ids, labels, filename: str):
    """
    Creates a submission file for Kaggle.
    """
    import pandas as pd

    assert len(ids) == len(labels)

    dataframe = pd.DataFrame(ids.ravel(), index=ids.ravel())
    dataframe['Prediction'] = labels
    del dataframe[0]

    dataframe.to_csv(filename, cols=["Id", "Prediction"])
