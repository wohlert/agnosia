"""
This is an example pipeline for submitting a solution
for the competition.
"""
import numpy as np

import megio as io
from preprocessing import cut_samples
import nn

folder = "../MEG"

X_train, X_test, y_train, y_test = io.load_subjects(folder, False)
sfreq, tmin, _ = io.load_meta(folder)

# Prepare the data
onset = int(abs(sfreq*tmin))
(X_train, X_test) = (cut_samples(X_train, onset), cut_samples(X_test, onset))

# Do training
(trials, channels, samples) = X_train.shape
model = nn.create_cnn(channels, samples)

# Classify
batch_size = 16
nb_epoch = 10

model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True)

# Create submission
prediction = model.predict_classes(X_test, batch_size=batch_size,
                                   show_accuracy=True, verbose=1)

io.create_submission(y_test, prediction, "submission_cnn.csv")
