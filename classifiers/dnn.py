"""
network.py

Provides different network models.
"""
import numpy as np
np.random.seed(1337)

from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Merge, Reshape, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score


def create_single_frame(input_shape):
    """
    Creates a CNN for a single image frame.
    """
    model = Sequential()

    # 4 32*3*3 convolution layers
    model.add(Convolution2D(32, 3, 3, border_mode="valid", input_shape=input_shape))
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


def functional_model(image_shape, frames):

    conv_input = Input(shape=image_shape)

    # 3 32*3*3 convolution layers
    conv1 = Convolution2D(32, 3, 3, border_mode="valid", activation="relu")(conv_input)
    conv1 = Convolution2D(32, 3, 3, activation="relu")(conv1)
    conv1 = Convolution2D(32, 3, 3, activation="relu")(conv1)
    max1  = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # 2 64*3*3 convolution layers
    conv2 = Convolution2D(64, 3, 3, border_mode="valid", activation="relu")(max1)
    conv2 = Convolution2D(64, 3, 3, activation="relu")(conv2)
    max2  = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # 1 128*3*3 convolution layer
    conv3 = Convolution2D(128, 3, 3, border_mode="valid", activation="relu")(max2)
    max3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

    # Model for convolutional network
    convnet = Model(input=conv_input, output=max3)

    # 7 input layers for convnerts
    inputs = [Input(shape=image_shape) for _ in range(frames)]

    # 7 convnets
    convnets = [convnet(input) for input in inputs]

    merge_nets = merge(convnets, mode="concat")
    reshape = Reshape((128, 7))(merge_nets)
    lstm = LSTM(128, input_dim=frames, input_length=128, return_sequences=False, activation="tanh")(reshape)
    # dropout1 = Dropout(0.5)(lstm)
    dense1 = Dense(512, activation="relu")(lstm)
    # dropout2 = Dropout(0.5)(dense1)
    prediction = Dense(1, activation="sigmoid")(dense1)

    return Model(input=inputs, output=prediction)


# Load data
from utils import random_split

X_train, X_test, y_train, y_test = random_split("images/", 32, 7)
_, frames, channels, width, height = np.shape(X_train)

# Reshape to match CNN shapes
X_train = list(X_train.reshape(frames, -1, channels, width, height))
X_test = list(X_test.reshape(frames, -1, channels, width, height))
image_shape = (channels, width, height)

# Create model
model = functional_model(image_shape, frames)
model.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer="adam")

#SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Create callbacks
checkpoint = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5")
early_stop = EarlyStopping(patience=2)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

callbacks = [
    # checkpoint,
    # early_stop,
    LossHistory()]

# Fit model
batch_size = 32
nb_epochs = 10
history = model.fit(X_train, y_train.ravel(), batch_size=batch_size,
                    nb_epoch=nb_epochs, callbacks=callbacks)

# Evaluate model
prediction = model.predict(X_test, batch_size=batch_size)

from IPython import embed
embed()

accuracy = accuracy_score(prediction, y_test.ravel())
print(accuracy)


