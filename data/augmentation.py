"""Residual attenuation augmentation (optional experiment, not the main method).

For a training pair (x, y) the degradation is d = x - y. We make a milder input
    x' = y + A * d,   A(p) = a + (1 - a) * ((1 - t) + t * L(p))
where L is the normalised, blurred brightness of the clean image, a ~ U(min, max),
t ~ U(0.5, 1). Bright regions keep more of the degradation (as in the paper's
synthesis, Eq. 2), dark regions get less. 'uniform' uses the mean of A everywhere,
as a control with the same average strength. Target y is unchanged.
"""
import random

import cv2
import numpy as np


def attenuation_map(clean, rng, min_strength=0.4, max_strength=0.9, sigma=10):
    lum = (clean @ np.array([0.2126, 0.7152, 0.0722])).astype(np.float32)
    lum = cv2.GaussianBlur(lum, (0, 0), sigma)
    span = float(np.ptp(lum))
    light = (lum - lum.min()) / span if span > 1e-6 else np.full_like(lum, 0.5)
    a = rng.uniform(min_strength, max_strength)
    t = rng.uniform(0.5, 1.0)
    return (a + (1 - a) * ((1 - t) + t * light))[..., None]


def attenuate_residual(degraded, clean, mode='light', rng=None, min_strength=0.4, max_strength=0.9):
    """degraded, clean: float HxWx3 in [0, 1]. Returns the attenuated input."""
    if mode not in ('light', 'uniform'):
        raise ValueError(f'unknown mode {mode}')
    rng = rng or random.Random()
    A = attenuation_map(clean, rng, min_strength, max_strength)
    if mode == 'uniform':
        A = np.full_like(A, A.mean())
    return (clean + A * (degraded - clean)).astype(np.float32)
