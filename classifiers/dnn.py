"""
network.py

Provides different network models.
"""
import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Merge, Reshape
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score

import pandas as pd
from scipy.misc import imread

def create_single_frame(input_shape):
    """
    Creates a CNN for a single image frame.
    """
    from keras.layers import Convolution2D, MaxPooling2D

    model = Sequential()

    # 4 32*3*3 convolution layers
    model.add(Convolution2D(32, 3, 3, border_mode="valid", input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 2 64*3*3 convolution layers
    model.add(Convolution2D(64, 3, 3, border_mode="valid"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 1 128*3*3 convolution layer
    model.add(Convolution2D(128, 3, 3, border_mode="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    return model


def create_multi_frame(cnn_shape, frames):
    """
    Create 7 parallel CNNs that converge into a recurrent
    LSTM layer to make a prediction.
    """
    from keras.layers.recurrent import LSTM

    model = Sequential()

    # Create 7 CNNs and merge the outputs
    convnets = [create_single_frame(cnn_shape) for _ in range(frames)]
    model.add(Merge(convnets, mode="concat"))
    model.add(Reshape((128, frames)))

    # LSTM layer - only keep last prediction
    model.add(LSTM(128, input_dim=frames, input_length=128, return_sequences=False))
    model.add(Activation("tanh"))

    # Fully connected layer
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation("relu"))

    # Prediction layer
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    return model


def reformat_images(X, shape):
    X = X.astype("float32")
    X /= 255
    X = X.reshape(shape)
    return X


def random_split(base_directory, image_size):
    from sklearn.cross_validation import train_test_split

    frames = 7

    df = pd.read_csv("train_labels.csv")[:2000]

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


# Load data
X_train, X_test, y_train, y_test = random_split("images/", 32)
_, frames, channels, width, height = np.shape(X_train)

# Reshape to match CNN shapes
X_train = list(X_train.reshape(frames, -1, channels, width, height))
X_test = list(X_test.reshape(frames, -1, channels, width, height))
cnn_shape = (channels, width, height)

# Create model
model = create_multi_frame(cnn_shape, frames)
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True))

# Create callbacks
checkpoint = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5")
early_stop = EarlyStopping(patience=2)

# Fit model
batch_size = 20
nb_epochs = 5
history = model.fit(X_train, y_train.ravel(), batch_size=batch_size,
                    nb_epoch=nb_epochs, validation_split=0.2,
                    callbacks=[checkpoint, early_stop])

# Evaluate model
prediction = model.predict_classes(X_test, batch_size=batch_size)
accuracy = accuracy_score(prediction, y_test.ravel())
print(accuracy)

from IPython import embed

embed()

