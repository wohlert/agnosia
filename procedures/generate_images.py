import sys

import numpy as np
import pandas as pd

import atone.io as io
from atone.pipeline import Pipeline
from atone.preprocessing import scale, cut, get_magnetometers, keep_channels
from atone.imaging import windowed_wavelet_topography, generate_images, spatial_transforms


filename = str(sys.argv[1])

data_dir = "data/"

magnetometers = get_magnetometers("channel_names.npy")

# Load sensor map and create spatial transform
positions = io.load_positions(data_dir + "sensorspace.mat")
positions = positions[magnetometers]
coordinates = spatial_transforms(positions)

# Load subject data and metadata
X, y, names = io.load_subject(data_dir + filename)
sfreq, tmin, _ = io.load_meta(data_dir)
onset = sfreq * abs(tmin)

# Number of frames to split data into
frames = 7

pipe = Pipeline()
pipe.add(scale)
pipe.add(cut, [onset])
pipe.add(keep_channels, ["magnetometers"])
pipe.add(windowed_wavelet_topography, [frames])

X = pipe.run(X)

# Generate image files
# Example image file: "images/train_subject1/trial22.1.jpeg"
generate_images(X, coordinates, "images/", names, frames=frames, resolution=32)

# Generate class label file
# example filename: "train_subject1.csv"
label_filename = filename.split("/")[-1][:-3] + "csv"
labels = pd.DataFrame({"name": names, "label": y.ravel()})
labels.to_csv(label_filename, mode="w", index=False, header=None)

