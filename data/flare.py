# Synthetic flare-only pairs, following the paper's Eq. 1:  X_flare = alpha * X + beta * (L * K_APSF)
# L: light-source map by thresholding (soft edges instead of alpha matting)
# K_APSF: approximated by a narrow + wide Gaussian mixture
# beta: set from the light sources, so the glow peak has a random brightness

import cv2
import numpy as np

ALPHA = 0.995  # paper, Implementation Details


def _blur(img, sigma):
    if sigma <= 8:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    # wide blurs are smooth, so do them at 1/4 resolution (much faster)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 4)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def add_flare(clean, rng, min_peak=0.1, max_peak=0.5):
    """clean: float HxWx3 in [0, 1]. Returns the flare image, or None if there are no light sources."""
    lum = clean.mean(2)
    mask = (lum > rng.uniform(0.8, 0.95)).astype(np.float32)
    if mask.sum() < 4:
        return None
    mask = cv2.GaussianBlur(mask, (0, 0), 2)
    light = clean.astype(np.float32) * mask[..., None]

    w = rng.uniform(0.3, 0.7)
    glow = w * _blur(light, rng.uniform(4, 15)) + (1 - w) * _blur(light, rng.uniform(20, 60))
    beta = rng.uniform(min_peak, max_peak) / max(float(glow.mean(2).max()), 1e-6)
    return np.clip(ALPHA * clean + beta * glow, 0, 1).astype(np.float32)
