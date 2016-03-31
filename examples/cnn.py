"""
cnn

Uses a convolutional neural network to classify examples.
"""
import numpy as np
import agnosia.io as io
import agnosia.nn as nn
from agnosia.preprocessing import cut_samples, normalise, dropout_channels
from agnosia.pipeline import Pipeline

np.random.seed(8829)

folder = "../data"

# Load data and metadata
X_train, X_test, y_train, y_test = io.load_subjects(folder, False)
sfreq, tmin, _ = io.load_meta(folder)
onset = int(abs(sfreq*tmin))

# Create pipeline
pipeline = Pipeline()
pipeline.add(cut_samples, [onset])
pipeline.add(normalise)
pipeline.add(dropout_channels)

# Run pipeline
X_train = pipeline.run(X_train)
X_test = pipeline.run(X_test)

# Create classifier
_, channels, samples = X_train.shape
model = nn.create_cnn(channels, samples)

batch_size = 16
nb_epoch = 10
model.fit(X_train, y_train.ravel(), batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True)

# Create submission
prediction = model.predict_classes(X_test, batch_size=batch_size, \
                                   show_accuracy=True, verbose=1)

io.create_submission(y_test.ravel(), prediction, "submission_cnn.csv")
