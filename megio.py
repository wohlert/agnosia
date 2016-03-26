import numpy as np
from scipy.io import loadmat

def get_files(folder):
    """
    Retrieves a list of string of the filenames
    of .mat files in a folder.
    """
    import os

    train_path = "{}/train/".format(folder)
    train_files = list(filter(lambda f: f.endswith(".mat"), os.listdir(train_path)))
    train_files = ["{}{}".format(train_path, i) for i in train_files]

    test_path = "{}/test/".format(folder)
    test_files = list(filter(lambda f: f.endswith(".mat"), os.listdir(test_path)))
    test_files = ["{}{}".format(test_path, i) for i in test_files]

    return train_files, test_files

def load_subjects(folder, no_of_subjects=False):
    """
    Loads a number of subjects and splits the data accordingly.
    """
    from sklearn.cross_validation import train_test_split

    train_files, test_files = get_files(folder)

    # No subjects chosen, do true split
    if not no_of_subjects:
        train_subjects = [loadmat(x) for x in train_files]
        test_subjects = [loadmat(x) for x in test_files]

        X_train = np.vstack([subject['X'] for subject in train_subjects])
        y_train = np.vstack([subject['y'] for subject in train_subjects])

        X_test = np.vstack([subject['X'] for subject in test_subjects])
        y_test = np.vstack([subject['Id'] for subject in test_subjects])

    # Do a split for cross validation
    else:
        train_subjects = [loadmat(x) for x in train_files[:no_of_subjects]]
        X = np.vstack([subject['X'] for subject in train_subjects])
        y = np.vstack([subject['y'] for subject in train_subjects])

        X_train, X_test, y_train, y_test = train_test_split(X, y)

    return X_train, X_test, y_train, y_test

def load_meta(folder):
    """
    Loads the meta data for a .mat files in a folder.
    """
    train_files, _ = get_files(folder)
    subject = loadmat(train_files[0])

    sfreq = int(subject['sfreq'])
    tmin = int(subject['tmin'])
    tmax = int(subject['tmax'])

    return sfreq, tmin, tmax

def create_submission(ids, labels, filename):
    """
    Creates a submission file for Kaggle.
    """
    import pandas as pd

    assert len(ids) == len(labels)

    df = pd.DataFrame(ids.ravel(), index=ids.ravel())
    df['Prediction'] = labels
    del df[0]

    df.to_csv(filename, cols=["Id", "Prediction"])
