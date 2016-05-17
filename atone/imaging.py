"""
imaging

Subroutines for creating images
in the style from EEGLearn.
"""

import numpy as np
from numpy.fft import rfft
from scipy.interpolate import CloughTocher2DInterpolator
from PIL import Image
import pywt as wave

from .constants import DELTA, THETA, ALPHA
from .preprocessing import min_max
from .frequency import bandpass, fft


def _cartesian_to_spherical(x: float, y: float, z: float) -> tuple:
    """
    Transform Cartesian coordinates to spherical.
    """
    x2_y2 = x**2 + y**2
    radius = np.sqrt(x2_y2 + z**2)
    elevation = np.arctan2(z, np.sqrt(x2_y2))
    azimuth = np.arctan2(y, x)
    return radius, elevation, azimuth


def _polar_to_cartesian(theta: float, rho: float) -> tuple:
    """
    Transform polar coordinates to Cartesian.
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def spatial_transforms(positions: np.array) -> np.array:
    """
    Converts multiple coordinates from 3D to 2D space.
    """
    def spatial_transform(x: float, y: float, z: float) -> np.array:
        [_, elevation, azimuth] = _cartesian_to_spherical(x, y, z)
        return _polar_to_cartesian(azimuth, np.pi / 2 - elevation)

    return np.array([spatial_transform(*positions[i]) for i in range(len(positions))])


def windowed_spectral_topography(input_matrix: np.array, frames: int=7, normalisation=None) -> np.array:
    """
    Generates a series of 3-dimensional spectral topographies
    over the number of channels.

    Creates a number of time windows based on the samples and
    applies the fourier transform to this data, effectievly
    applying short-time fourier transform (STFT) on the data.
    """
    trials, channels, samples = np.shape(input_matrix)

    if frames < 2:
        transforms = [fft(input_matrix, lower_limit=1)]
    else:
        frame_length = int(samples/frames)
        frame_list = [frame_length*i for i in range(1, frames+1)]
        frame_ranges = [range(frame_list[i] - frame_length, frame_list[i]) for i in range(frames)]

        transforms = [fft(input_matrix[:, :, f], lower_limit=1) for f in frame_ranges]

    if not normalisation:
        normalisation = lambda x: x

    def band_extraction(band, transform):
        lower, upper = band
        subband = np.mean(transform[:, :, lower:upper], axis=-1)
        subband = normalisation(subband)

        return subband

    bands = (DELTA, THETA, ALPHA)
    return np.dstack([band_extraction(band, transform) for band in bands for transform in transforms]).reshape(-1, channels, 3)


def spectral_topography(input_matrix: np.array, normalisation=None) -> np.array:
    """
    Generates a single 3-dimensional spectral topography
    over the number of channels.
    """
    return windowed_spectral_topography(input_matrix, frames=1, normalisation=normlisation)


def windowed_wavelet_topography(input_matrix: np.array, frames: int=7):
    """
    Generates a single 3-dimensional spectral topography
    over the number of channels.

    Uses Daubechies 4 wavelet and 5 levels of decomposition
    for a discrete wavelet transform (DWT) on the data in
    order to get approximately these bands:

    let Fm be 0.5 * sampling_frequency

     0 -   4Hz : 0 - fm/32      ~ approximation1
     4 -   8Hz : fm/32 - fm/16  ~ detail5
     8 -  16Hz : fm/16 - fm/8   ~ detail4
    16 -  32Hz : fm/8 - fm4     ~ detail3
    32 -  64Hz : fm/4 - fm2     ~ detail2
    64 - 125Hz : fm/2 - fm      ~ detail1

    Only the bottom 3 frequency bands are used to generate
    the topography due to the high frequency resolution.
    """
    wavelet = wave.Wavelet("db4")
    delta, theta, alpha, *_ = wave.wavedec(input_matrix, wavelet, level=5)

    frequency_bands = []

    for band in [delta, theta, alpha]:
        trials, channels, samples = np.shape(band)

        # Downsample and limit to `frames` number of samples
        division = samples // frames
        cut_band = band[:, :, ::division]
        cut_band = cut_band[:, :, :frames]

        # Normalise along samples axis
        normalised_band = np.apply_along_axis(min_max, 2, cut_band)

        frequency_bands.append(normalised_band)

    frequency_bands = np.array(frequency_bands)
    frequency_bands = np.reshape(frequency_bands, (-1, channels, 3))

    return frequency_bands


def interpolate_spectrum(value: np.array, coordinates: np.array, resolution: int=64) -> np.array:
    """
    Generates an image from a topography expressed in
    3 colour dimensions along with a 2D map of channels.
    """

    # Images should be square
    width = height = np.round(np.max(coordinates))
    step_size = (width + height) / resolution

    x, y = np.meshgrid(np.arange(-width, width, step_size), np.arange(-height, height, step_size))

    interpolation = CloughTocher2DInterpolator(coordinates, value, fill_value=0.0)
    z = interpolation.__call__(np.c_[x.ravel(), y.ravel()])

    img_width = img_height = int(np.sqrt(len(z)))

    img_data = z.reshape(img_width, img_height, 3)
    img = Image.fromarray((img_data * 255).astype(np.uint8))

    return img


def generate_images(values: np.array, coordinates: np.array, location: str, basenames: list, frames: int=1, resolution: int=64) -> None:
    """
    Generates and saves a set of images given
    their spectral topography, coordinates and
    save location.
    """
    if frames < 2:
        for i, value in enumerate(values):
            img = interpolate_spectrum(value, coordinates, resolution)

            filename = "".join([location, basenames[i]])
            img.save("{}.jpeg".format(filename), "JPEG")

    else:
        total_batch = int(len(values) / frames)

        for batch in range(total_batch):
            for frame in range(frames):
                value = values[batch+frame]
                img = interpolate_spectrum(value, coordinates, resolution)

                filename = "".join([location, basenames[batch], ".", str(frame)])
                img.save("{}.jpeg".format(filename), "JPEG")

