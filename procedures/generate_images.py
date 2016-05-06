import numpy as np
import sys

import atone.io as io
from atone.pipeline import Pipeline
from atone.preprocessing import scale, cut
from atone.frequency import Filterbank
from atone.imaging import generate_images, create_colors, create_colors2

X, names = io.load_subject("../data/" + str(sys.argv[1]))
sfreq, tmin, _ = io.load_meta("../data")
onset = sfreq * abs(tmin)

coordinates = np.load("sensormap.npy")

bank = Filterbank()

pipe = Pipeline()
pipe.add(scale)
pipe.add(cut, [onset])
pipe.add(bank.apply)
pipe.add(create_colors2)

X = pipe.run(X)

generate_images(X, coordinates, "images/", names, resolution=120)
