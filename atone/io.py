"""
io

Provides input and output operations for loading
MEG data - signals, sensorlocations and creating submissions.
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
    train_files = ["{}{}".format(train_path, f)
                   for f in os.listdir(train_path)
                   if f.endswith(".mat")]

    test_path = "{}/test/".format(folder)
    test_files = ["{}{}".format(test_path, f)
                  for f in os.listdir(test_path)
                  if f.endswith(".mat")]

    return train_files, test_files


def load_subjects(folder: str, no_of_subjects: int=0, cv: bool=False):
    """
    Loads a number of subjects and splits the data accordingly.
    """
    from sklearn.cross_validation import train_test_split

    train_files, test_files = get_files(folder)

    if cv:
        train_subjects = [loadmat(p) for p in train_files]

        x_train = np.vstack([subject['X'] for subject in train_subjects[:-1]])
        y_train = np.vstack([subject['y'] for subject in train_subjects[:-1]])

        x_test = np.vstack([subject['X'] for subject in test_subjects[-1]])
        y_test = np.vstack([subject['y'] for subject in test_subjects[-1]])

        return x_train, x_test, y_train.ravel(), y_test.ravel()

    # No subjects chosen, do true split
    if no_of_subjects < 1:
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

    return x_train, x_test, y_train.ravel(), y_test.ravel()


def load_subject(filepath: str):
    """
    Loads a single subject with names.
    """
    def get_subject(filename: str):
        subject_name = filename.split("/")[-1]
        head, *tail = subject_name.split(".")
        return head

    subject = loadmat(filepath)
    X = subject['X']

    try:
        # Training subject
        y = subject['y']
    except KeyError:
        # Test subject
        y = subject['Id']

    ids = np.arange(1, len(X) + 1)

    name = get_subject(filepath)
    names = np.array(["{}/trial{}".format(name, i) for i in ids])

    return X, y.ravel(), names


def load_meta(folder: str):
    """
    Loads the metadata for a signle .mat files.
    """
    train_files, _ = get_files(folder)
    subject = loadmat(train_files[0])

    sfreq = float(subject['sfreq'])
    tmin = float(subject['tmin'])
    tmax = float(subject['tmax'])

    return sfreq, tmin, tmax


def load_positions(filepath: str):
    """
    Loads sensor location .mat file.
    """
    sensor_placement = loadmat(filepath)
    positions = sensor_placement['pos']

    return positions


def create_submission(ids, labels, filename: str):
    """
    Creates a submission file for Kaggle.
    """
    import pandas as pd

    assert len(ids) == len(labels), "Length of y-labels must be the same as for Ids"

    dataframe = pd.DataFrame(ids, index=ids)
    dataframe['Prediction'] = labels
    del dataframe[0]

    dataframe.to_csv(filename, cols=["Id", "Prediction"])

