"""
wavelet
"""


import numpy as np
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import atone.io as io
from atone.preprocessing import scale, cut
from atone.frequency import bandpass, dwt_summary, dwt_spectrum
from atone.features import pool
from atone.pipeline import Pipeline

np.random.seed(8829)

folder = "data"

# Load data and metadata
subjects = 1

X_train, X_test, y_train, y_test = io.load_subjects(folder, no_of_subjects=subjects)
sfreq, tmin, _ = io.load_meta(folder)
onset = int(abs(sfreq*tmin))

# Create pipeline
pipeline = Pipeline()
pipeline.add(scale)
pipeline.add(cut, [onset])
pipeline.add(bandpass, [sfreq, 0.25, 25])  # Best bandpass based on cross validation
pipeline.add(dwt_spectrum)
pipeline.add(pool)

# Run pipeline
X_train = pipeline.run(X_train)
X_test = pipeline.run(X_test)

_, size = np.shape(X_train)

# Create classifier
model = Sequential()
model.add(Dense(512, input_dim=size))
model.add(Activation('linear'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('linear'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd')

model.fit(X_train, y_train.ravel(),
          batch_size=32, nb_epoch=5,
          verbose=1)

prediction = model.predict_classes(X_test, batch_size=32)

# model = LogisticRegression()
# model.fit(X_train, y_train.ravel())

# prediction = model.predict(X_test)

print(accuracy_score(prediction, y_test.ravel()))
