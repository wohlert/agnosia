"""
pooling

This is an example pipeline demonstrating the creation
of a classifier based on trial-wise pooling along
with a support vector machine classifier on the data.

This particular solution will yield an accuracy of
0.69 on the competition.
"""


import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm

import atone.io as io
from atone.preprocessing import normalise, cut
from atone.features import pool
from atone.pipeline import Pipeline

np.random.seed(8829)

folder = "../data"

# Load data and metadata
subjects = 0

if int(sys.argv[1]):
    subjects = int(sys.argv[1])

X_train, X_test, y_train, y_test = io.load_subjects(folder, no_of_subjects=subjects)
sfreq, tmin, _ = io.load_meta(folder)
onset = int(abs(sfreq*tmin))

# Create pipeline
pipeline = Pipeline()
pipeline.add(normalise)
pipeline.add(cut, [onset])
pipeline.add(pool)

# Run pipeline
X_train = pipeline.run(X_train)
X_test = pipeline.run(X_test)

# Create classifier
model = svm.SVC()
model.fit(X_train, y_train.ravel())

# Create submission
prediction = model.predict(X_test)

if subjects:
    print(accuracy_score(prediction, y_test.ravel()))
else:
    io.create_submission(y_test.ravel(), prediction, "submission_pooling.csv")
