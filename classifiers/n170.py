"""
M170

Uses the concept of the 170ms ERP to faces
along with gamma wave band isolation to
classify samples.
"""
import numpy as np

import atone.io as io
import atone.preprocessing as pre
from atone.pipeline import Pipeline

seed = np.random.seed(8829)

folder = "../data"

# Load data and metadata
X_train, X_test, y_train, y_test = io.load_subjects(folder, no_of_subjects=1)
sfreq, tmin, tmax = io.load_meta(folder)

# Find the channels to drop
channel_pipe = Pipeline()
channel_pipe.add(pre.normalise)
channel_pipe.add(pre.dropout_channels_tanh)
channels = channel_pipe.run(X_train)

# Create pipeline
pipeline = Pipeline()
pipeline.add(pre.normalise)
pipeline.add(pre.cut_m170, [sfreq, tmin, 5])
pipeline.add(np.trapz)

# Run pipeline
X_train = pipeline.run(X_train)
X_test = pipeline.run(X_test)

# Train classifier
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train.ravel())

# Make prediction
prediction = model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(prediction, y_test.ravel()))
