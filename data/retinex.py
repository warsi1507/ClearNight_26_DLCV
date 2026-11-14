# Single-scale Retinex decomposition used to precompute the two priors.
# Same as the official Retinex.py (BSD-3-Clause), plus a guard for flat images.
#   illumination = Gaussian blur of the image          -> saved in <subset>1/
#   reflectance  = log10(I + 1) - log10(blur(I) + 1)    -> saved in <subset>2/
# Both are min-max normalised to [0, 255] per image.

import os

import cv2
import numpy as np


def single_scale_retinex(img, sigma):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log10(img + 1) - np.log10(blurred + 1)
    return retinex, blurred


def normalize_map(values):
    span = float(np.ptp(values))
    if span <= 1e-8:  # flat image, avoid 0/0
        return np.zeros_like(values)
    return (values - values.min()) / span * 255


def decompose(img, sigma=10):
    """Returns (reflectance, illumination) as uint8 images."""
    img = img.astype(np.float32) / 255.0
    reflectance, illumination = single_scale_retinex(img, sigma)
    return normalize_map(reflectance).astype(np.uint8), normalize_map(illumination).astype(np.uint8)


def process_folder(input_dir, reflectance_dir, illumination_dir, sigma=10):
    os.makedirs(reflectance_dir, exist_ok=True)
    os.makedirs(illumination_dir, exist_ok=True)
    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img = cv2.imread(os.path.join(input_dir, filename))
        if img is None:
            raise ValueError(f'cannot read {os.path.join(input_dir, filename)}')
        reflectance, illumination = decompose(img, sigma)
        cv2.imwrite(os.path.join(reflectance_dir, filename), reflectance)
        cv2.imwrite(os.path.join(illumination_dir, filename), illumination)
