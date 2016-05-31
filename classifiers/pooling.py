"""
pooling

This is an example pipeline demonstrating the creation
of a classifier based on trial-wise pooling along
with a support vector machine classifier on the data.
"""
import numpy as np
from sklearn.svm import SVC
from atone.io import load_meta
from utils import run_cv

np.random.seed(8829)
sfreq, tmin, _ = load_meta("data")
onset = int(abs(sfreq*tmin))


def pipeline():
    from atone.pipeline import Pipeline
    from atone.preprocessing import cut, scale
    from atone.features import pool

    pipeline = Pipeline()
    pipeline.add(scale)
    pipeline.add(cut, [onset])
    pipeline.add(pool)
    return pipeline


pipe = pipeline()
model = SVC()
config = {"model": model, "pipeline": pipe, "subjects": 1}
run_cv(config, **config)

