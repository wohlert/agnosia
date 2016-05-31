import numpy as np
import pandas as pd
from scipy.misc import imread
import atone.io as io
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import logging
logging.basicConfig(filename='scores.log', level=logging.DEBUG)

def reformat_images(X, shape):
    X = X.astype("float32")
    X /= 255
    X = X.reshape(shape)
    return X


def random_split(base_directory, image_size, frames=7):
    from sklearn.cross_validation import train_test_split

    df = pd.read_csv("train_labels.csv")[:594]

    X = np.array([imread("{}{}.{}.jpeg".format(base_directory, i, f)) for i in df['name'].values for f in range(frames)])
    X = reformat_images(X, (-1, frames, 3, image_size, image_size))

    y = df['label'].values

    return train_test_split(X, y)


def loo_split(base_directory, image_size):
    frames = 7

    df = pd.read_csv("train_labels.csv")
    test_name = "train_subject{}".format(np.random.randint(1, 17))

    train_subjects = df[~df['name'].str.contains(test_name)]
    test_subject = df[df['name'].str.contains(test_name)]

    X_train = np.array([imread("{}{}.{}.jpeg".format(base_directory, i, f)) for i in train_subjects['name'].values for f in range(frames)])
    X_train = reformat_images(X_train, (-1, frames, 3, image_size, image_size))

    y_train = train_subjects['label'].values

    X_test = np.array([imread("{}{}.{}.jpeg".format(base_directory, i, f)) for i in test_subject['name'].values for f in range(frames)])
    X_test = reformat_images(X_test, (-1, frames, 3, image_size, image_size))

    y_test = test_subject['label'].values

    return X_train, X_test, y_train, y_test


def true_split(base_directory, image_size):
    frames = 7

    train_df = pd.read_csv("train_labels.csv")
    train_names = train_df['name'].values.astype(str)

    test_df = pd.read_csv("test_labels.csv")
    test_names = train_df['name'].values.astype(str)

    X_train = np.array([imread("{}{}.{}.jpeg".format(base_directory, i, f)) for i in train_names for f in range(frames)])
    X_train = reformat_images(X_train, (-1, frames, 3, image_size, image_size))

    y_train = train_df['label'].values

    X_test = np.array([imread("{}{}.{}.jpeg".format(base_directory, i, f)) for i in test_names for f in range(frames)])
    X_test = reformat_images(X_test.reshape, (-1, frames, 3, image_size, image_size))

    names = test_df['label'].values

    return X_train, X_test, y_train, names


def run_loo(config, pipeline, model):
    mean_score = 0

    X_train, X_test, y_train, y_test = io.load_subjects("data", cv=True)

    # Run pipeline
    X_train_pre = pipeline.run(X_train)
    X_test_pre = pipeline.run(X_test)

    # Fit model to data
    model.fit(X_train_pre, y_train)

    y_pred = model.predict(X_test_pre)
    score = accuracy_score(y_test, y_pred)

    logging.info(score)
    logging.info(config)



def run_cv(config, pipeline, model, subjects: int=1):
    mean_score = 0

    for i in range(subjects):
        X, y, _ = io.load_subject("data/train/train_subject{}.mat".format(i+1))
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Run pipeline
        X_train_pre = pipeline.run(X_train)
        X_test_pre = pipeline.run(X_test)

        # Fit model to data
        model.fit(X_train_pre, y_train)

        y_pred = model.predict(X_test_pre)
        score = accuracy_score(y_test, y_pred)
        print(confusion_matrix(y_test, y_pred))

        mean_score += score

    mean_score /= subjects

    logging.info(mean_score)
    logging.info(config)

