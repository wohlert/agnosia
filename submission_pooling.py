"""
This is an example pipeline for submitting a solution
for the competition.
"""
import numpy as np
from sklean import svm

import megio as io
from preprocessing import cut_samples
from pooling import pool

folder = "../MEG"

X_train, X_test, y_train, y_test = io.load_subjects(folder, False)
sfreq, tmin, _ = io.load_meta(folder)

# Prepare the data
onset = int(abs(sfreq*tmin))
(X_train, X_test) = (cut_samples(X_train, onset), cut_samples(X_test, onset))
(X_train, X_test) = (pool(X_train), pool(X_test))

# Classify
model = svm.SVC()
model.fit(X_train)

# Create submission
prediction = model.predict(X_test)

io.create_submission(y_test, prediction, "submission_pooling.csv")
