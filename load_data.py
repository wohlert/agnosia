import os
import numpy as np
from scipy.io import loadmat

def load(folder, subjects=None):
    train_path = "{}/train/".format(folder)
    train_files = list(filter(lambda f: f.endswith(".mat"), os.listdir(train_path)))

    test_path = "{}/test/".format(folder)
    test_files = list(filter(lambda f: f.endswith(".mat"), os.listdir(test_path)))

    if not subjects:
        train_subjects = [loadmat(train_path + x) for x in train_files]
        test_subjects = [loadmat(test_path + x) for x in test_files]

        X_train = np.vstack([subject['X'] for subject in train_subjects])
        y_train = np.vstack([subject['y'] for subject in train_subjects])

        X_test = np.vstack([subject['X'] for subject in test_subjects])
        y_test = np.vstack([subject['y'] for subject in test_subjects])

    else:
        from sklearn.cross_validation import train_test_split
        train_subjects = [loadmat(train_path + i) for i in train_files[:subjects]]
        X = np.vstack([subject['X'] for subject in train_subjects])
        y = np.vstack([subject['y'] for subject in train_subjects])

        X_train, X_test, y_train, y_test = train_test_split(X, y)

    return X_train, X_test, y_train, y_test

def get_nyquist(folder):
    subject_path = "{}/train/train_subject1.mat".format(folder)
    subject = loadmat(subject_path)

    sampling_freq = int(subject['sfreq'])
    nyquist = sampling_freq >> 1
    return nyquist
