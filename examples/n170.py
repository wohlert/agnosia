"""
N170

Uses the concept of the 170ms ERP to faces
along with gamma wave band isolation to
classify samples.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
import agnosia.io as io
from agnosia.preprocessing import cut_samples, dropout_channels
from agnosia.filterbank import full_filter_bank
from agnosia.pipeline import Pipeline

seed = np.random.seed(8829)

folder = "../data"

# Load data and metadata
X_train, X_test, y_train, y_test = io.load_subjects(folder)
sfreq, tmin, tmax = io.load_meta(folder)

# Find optimal temporal window around 170ms
trial_start = int(abs(tmin*sfreq))
prime = trial_start + 170
nmin = prime - 50
nmax = prime + 50

# Only use gamma banding
bands = [(25, 39), (40, 54), (55, 69), (70, 84), (85, 100)]

# Create pipeline
pipeline = Pipeline()
pipeline.add(cut_samples, [nmin, nmax])
pipeline.add(dropout_channels, [0.1])
pipeline.add(full_filter_bank, [bands])

# Run pipeline
print("Running pipeline on train")
X_train = pipeline.run(X_train)
print("Running pipeline on test")
X_test = pipeline.run(X_test)

print("Saving arrays for later")
np.save("training.npy", X_train)
np.save("testing.npy", X_test)

# Create classifier
print("Training classifier")
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train.ravel())

# Create submission
print("Generating submission")
prediction = model.predict(X_test)
io.create_submission(y_test, prediction, "submission_n170.csv")
