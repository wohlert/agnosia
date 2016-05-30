"""
bandpass

"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from atone.io import load_meta
from utils import run_loo


np.random.seed(8829)
sfreq, tmin, _ = load_meta("data")
onset = int(abs(sfreq*tmin))

def pipeline():
    from atone.pipeline import Pipeline
    from atone.preprocessing import scale, cut, keep_channels
    from atone.frequency import downsample, bandpass
    from atone.features import pool

    pipeline = Pipeline()
    pipeline.add(scale)
    pipeline.add(cut, [onset])
    pipeline.add(keep_channels, ["magnetometers"])
    pipeline.add(downsample, [2])
    pipeline.add(bandpass, [sfreq, 1.25, 25])
    pipeline.add(pool)
    return pipeline


pipe = pipeline()
model = LogisticRegression(penalty="l1", C=0.1)
config = {"model": model, "pipeline": pipe}
run_loo(config, **config)