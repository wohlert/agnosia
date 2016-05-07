import numpy as np
import sys
from scipy.stats import zscore

import atone.io as io
from atone.pipeline import Pipeline
from atone.preprocessing import scale, cut, normalise
from atone.imaging import generate_images, spectral_topography

X, names = io.load_subject("../data/" + str(sys.argv[1]))
sfreq, tmin, _ = io.load_meta("../data")
onset = sfreq * abs(tmin)

coordinates = np.load("sensormap.npy")

pipe = Pipeline()
pipe.add(scale)
pipe.add(cut, [onset])
pipe.add(normalise, [0])
pipe.add(spectral_topography)

X = pipe.run(X)

generate_images(X, coordinates, "images/", names, resolution=100)
