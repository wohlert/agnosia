"""
# Pooling

This is an example pipeline demonstrating the creation
of a classifier based on trial-wise pooling along
with a support vector machine classifier on the data.

This particular solution will yield an accuracy of
0.69 on the competition.
"""
import numpy as np
from sklean import svm
np.random.seed(8829)

import agnosia.io as io
from agnosia.preprocessing import cut_samples
from agnosia.features import pool
from agnosia.pipeline import Pipeline

folder = "../data"

# Load data and metadata
X_train, X_test, y_train, y_test = io.load_subjects(folder, False)
sfreq, tmin, _ = io.load_meta(folder)
onset = int(abs(sfreq*tmin))

# Create pipeline
pipeline = Pipeline()
pipeline.add(cut_samples, [onset]))
pipeline.add(pool)

# Run pipeline
X_train = pipeline.run(X_train)
X_test = pipeline.run(X_test)

# Create classifier
model = svm.SVC()
model.fit(X_train)

# Create submission
prediction = model.predict(X_test)
io.create_submission(y_test, prediction, "submission_pooling.csv")
