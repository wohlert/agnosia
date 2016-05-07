"""
imaging

Subroutines for creating images
in the style from EEGLearn.
"""

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from PIL import Image

from .frequency import fft
from .constants import DELTA, THETA, ALPHA


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


def spectral_topography(input_matrix: np.array, normalisation=None) -> np.array:
    """
    Generates a 3-dimensional spectral topography
    over the number of channels for use in imaging.
    """
    trials, channels, samples = np.shape(input_matrix)

    transform = fft(input_matrix, lower_limit=1)

    if not normalisation:
        normalisation = lambda x: x

    def band_extraction(band):
        lower, upper = band
        subband = np.mean(transform[:, :, lower:upper], axis=-1)
        subband = normalisation(subband)

        return subband

    bands = (DELTA, THETA, ALPHA)
    return np.dstack([band_extraction(band) for band in bands]).reshape(-1, channels, 3)


def spectral_topography_window(input_matrix: np.array, windows: int, normalisation=None) -> np.array:
    pass


def generate_images(values: np.array, coordinates: np.array, location: str, basenames: list, resolution: int=60):
    """
    Generates a set of images from spectral topography
    and 2D map of channels.
    """

    # Images should be square
    width = height = np.round(np.max(coordinates))
    step_size = (width + height) / resolution

    x, y = np.meshgrid(np.arange(-width, width, step_size), np.arange(-height, height, step_size))

    for i, value in enumerate(values):
        interpolation = CloughTocher2DInterpolator(coordinates, value, fill_value=0.0)
        z = interpolation.__call__(np.c_[x.ravel(), y.ravel()])

        img_width = img_height = int(np.sqrt(len(z)))

        img_data = z.reshape(img_width, img_height, 3)
        img = Image.fromarray((img_data * 255).astype(np.uint8))

        filename = "".join([location, basenames[i]])
        img.save("{}.jpeg".format(filename), "JPEG")
