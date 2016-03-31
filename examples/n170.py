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

folder = "../MEG"

# Load data and metadata
X_train, X_test, y_train, y_test = io.load_subjects(folder)
sfreq, tmin, tmax = io.load_meta(folder)

# Find optimal temporal window around 170ms
trial_start = int(abs(tmin*sfreq))
prime = trial_start + (sfreq*0.170)
nmin = int(prime - sfreq*0.050)
nmax = int(prime + sfreq*0.050)

# Only use gamma banding

# Create pipeline
pipeline = Pipeline()
pipeline.add(cut_samples, [nmin, nmax])
#pipeline.add(dropout_channels, [0.1])
pipeline.add(full_filter_bank)

# Run pipeline
X_train = pipeline.run(X_train)
X_test = pipeline.run(X_test)

model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train.ravel())

prediction = model.predict(X_test)
io.create_submission(y_test.ravel(), prediction, "submission_n170.csv")
