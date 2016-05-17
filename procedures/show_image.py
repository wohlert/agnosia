import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import atone.io as io
from atone.pipeline import Pipeline
from atone.preprocessing import scale, cut
from atone.imaging import interpolate_spectrum, windowed_wavelet_topography, spatial_transforms


filename = str(sys.argv[1])
frames = int(sys.argv[2])

data_dir = "data/"

# Load sensor map and create spatial transform
positions = io.load_positions(data_dir + "sensorspace.mat")
coordinates = spatial_transforms(positions)

# Load subject data and metadata
X, y, names = io.load_subject(data_dir + filename)
sfreq, tmin, _ = io.load_meta(data_dir)
onset = sfreq * abs(tmin)

# Number of frames to split data into
pipe = Pipeline()
pipe.add(scale)
pipe.add(cut, [onset])
pipe.add(windowed_wavelet_topography, [frames])

X = pipe.run(X)

X = X.reshape(-1, frames, 306, 3)

for i in range(frames):
    img = interpolate_spectrum(X[0, i], coordinates, resolution=128)
    plt.imshow(img)
    plt.show()

