import numpy as np

import atone.io as io
from atone.imaging import spatial_transforms

positions = io.load_positions("../data/sensorspace.mat")
coordinates = spatial_transforms(positions)

np.save("sensormap.npy", coordinates)
