"""
network.py

Provides different network models.
"""

import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score


def create_single_frame(input_shape):
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

    # 512 neuron fully-connected layer
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))

    # Output layer
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def load_data(base):
    from scipy.misc import imread
    from sklearn.cross_validation import train_test_split

    y = np.load("labels.npy")

    X = np.array([imread("{}/trial{}.jpeg".format(base, i)) for i in np.arange(1, len(y)+1)])
    X = X.astype('float32')
    X /= 255

    return train_test_split(X, y)


input_shape = (3, 120, 120)

model = create_single_frame(input_shape)

batch_size = 20
nb_epochs = 2

# Create model
model = create_single_frame(input_shape)

optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='binary_crossentropy', optimizer='sgd')

X_train, X_test, y_train, y_test = load_data("images/train_subject1")

X_train = X_train.reshape(-1, 3, 120, 120)
X_test = X_test.reshape(-1, 3, 120, 120)

# Fit model
model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epochs)

# Evaluate model
prediction = model.predict_classes(X_test, batch_size=batch_size)
accuracy = accuracy_score(prediction, y_test)
print(accuracy)
