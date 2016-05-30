import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Merge, Reshape, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score

from .utils import loo_split

def create_single_frame(input_shape):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))

    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


# Load data
X_train, X_test, y_train, y_test = loo_split("bw/", 100)
_, channels, width, height = np.shape(X_train)

X_train = X_train[:1000]
y_train = y_train[:1000]

X_test = X_test[:100]
y_test = y_test[:100]

# Reshape to match CNN shapes
cnn_shape = (1, width, height)

# Create model
model = create_single_frame(cnn_shape)
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

# Fit model
batch_size = 32
nb_epochs = 5
history = model.fit(X_train, y_train.ravel(),
                    batch_size=batch_size,
                    nb_epoch=nb_epochs)

# Evaluate model
prediction = model.predict_classes(X_test, batch_size=batch_size)
accuracy = accuracy_score(prediction, y_test.ravel())
print(accuracy)

