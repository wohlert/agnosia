import numpy as np
import sys
import pandas as pd

import atone.io as io
from atone.pipeline import Pipeline
from atone.preprocessing import scale, cut, normalise
from atone.imaging import generate_images, spectral_topography

X, y, names = io.load_subject("../data/" + str(sys.argv[1]))
sfreq, tmin, _ = io.load_meta("../data")
onset = sfreq * abs(tmin)

split = str(sys.argv[1]).split("/")
filename = split[-1][:-3] + "csv"
pd.DataFrame({"name": names, "label": y.ravel()}).to_csv(filename, mode="w", index=False, header=None)

coordinates = np.load("sensormap.npy")

pipe = Pipeline()
pipe.add(scale)
pipe.add(cut, [onset])
pipe.add(normalise, [0])
pipe.add(spectral_topography)

X = pipe.run(X)

generate_images(X, coordinates, "images/", names, resolution=100)
