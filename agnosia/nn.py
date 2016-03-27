from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

def create_rnn(input_length):
    from keras.layers import LSTM
    """
    Uses a deep neural network with LSTM activation
    to classify the data.

    Topology:

        * X

    """
    model = Sequential()
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    return model

def create_cnn(channels, samples):
    from keras.layers.convolution import Convolution1D, MaxPooling1D
    from keras.layers.advanced_activations import PReLU
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
    model = Sequential()
    model.add(Convolution1D(nb_filter, filter_length, border_mode='valid', input_shape=(channels, samples)))
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
