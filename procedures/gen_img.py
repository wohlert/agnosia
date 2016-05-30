import sys

import numpy as np
import pandas as pd

import atone.io as io
from atone.pipeline import Pipeline
from atone.preprocessing import scale, cut, get_magnetometers, min_max, keep_channels
from atone.imaging import spatial_transforms, generate_images


filename = str(sys.argv[1])

data_dir = "data/"

magnetometers = get_magnetometers("./channel_names.npy")

# Load sensor map and create spatial transform
positions = io.load_positions(data_dir + "sensorspace.mat")
positions = positions[magnetometers]
coordinates = spatial_transforms(positions)

# Load subject data and metadata
X, y, names = io.load_subject(data_dir + filename)
sfreq, tmin, _ = io.load_meta(data_dir)
onset = sfreq * abs(tmin)


def erp_topography(input_matrix: np.array):
    trials, channels, samples = input_matrix.shape
    n170 = int(sfreq*0.170)
    output = input_matrix[:, :, n170]
    return output.reshape(trials, channels)


# Number of frames to split data into
pipe = Pipeline()
pipe.add(scale)
pipe.add(cut, [onset])
pipe.add(keep_channels, [magnetometers])
pipe.add(erp_topography)

X = pipe.run(X)
X = np.apply_along_axis(min_max, 1, X)

# Generate image files
# Example image file: "images/train_subject1/trial22.1.jpeg"
generate_images(X, coordinates, "bw/", names, resolution=100)

