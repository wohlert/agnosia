"""
wavelet

"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from atone.io import load_meta
from utils import run_cv


np.random.seed(8829)
sfreq, tmin, _ = load_meta("data")
onset = int(abs(sfreq*tmin))

def pipeline():
    from atone.pipeline import Pipeline
    from atone.preprocessing import scale, cut
    from atone.frequency import dwt_spectrum
    from atone.features import pool

    pipeline = Pipeline()
    pipeline.add(scale)
    pipeline.add(cut, [onset])
    pipeline.add(dwt_spectrum, [5, "db2", (0, 2, 3)])
    pipeline.add(pool)
    return pipeline


pipe = pipeline()
model = LogisticRegression(penalty="l1", C=0.1)
config = {"model": model, "pipeline": pipe, "subjects": 16}
run_cv(config, **config)

