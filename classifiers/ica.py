"""
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from atone.io import load_meta
from utils import run_cv

np.random.seed(8829)
sfreq, tmin, _ = load_meta("data")
onset = int(abs(sfreq*tmin))


def pipeline():
    from atone.pipeline import Pipeline
    from atone.preprocessing import cut, scale, keep_channels
    from atone.features import ica, pool

    pipeline = Pipeline()
    pipeline.add(scale)
    pipeline.add(cut, [onset])
    pipeline.add(keep_channels, ["magnetometers"])
    pipeline.add(ica)
    pipeline.add(pool)
    return pipeline


pipe = pipeline()
model = LogisticRegression(penalty="l1", C=0.1)
config = {"pipeline": pipe, "model": model, "subjects": 1}

run_cv(config, **config)

