"""
nn

Neural network implementations with standard hyperparameters
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

def create_rnn(input_length: int):
    """
    Uses a deep neural network with LSTM activation
    to classify the data.

    Topology:

        * X

    """
    from keras.layers import LSTM

    model = Sequential()
    model.add(LSTM(output_dim=input_length, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    return model

def create_cnn(channels: int, samples: int) -> Sequential:
    """
    Creates a convolutional neural network to classify data.

    Topology:

        * Convolution [input]
        * PReLu
        * Convolution
        * PReLU
        * Maxpooling
        * Dropout
        * Convolution
        * PReLu
        * Maxpooling
        * Flatten
        * Fully connected
        * Dropout
        * Fully connected [output]

    """
    from keras.layers.convolution import Convolution1D, MaxPooling1D
    from keras.layers.advanced_activations import PReLU

    nb_filter = 64
    filter_length = 3
    pool_length = 2

    model = Sequential()
    model.add(Convolution1D(nb_filter, filter_length, border_mode='valid', \
                            input_shape=(channels, samples)))
    model.add(PReLU())

    model.add(Convolution1D(nb_filter, filter_length, border_mode='valid'))
    model.add(PReLU())

    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(Dropout(0.5))

    model.add(Convolution1D(nb_filter, filter_length, border_mode='valid'))
    model.add(PReLU())

    model.add(MaxPooling1D(pool_length=pool_length))

    model.add(Flatten())
    model.add(Dense(nb_filter))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')

    return model
