# PSNR / SSIM. Based on the official utils/UTILS.py (BSD-3-Clause).
#
# protocol='native'   : clamp to [0, 1], float RGB PSNR, Gaussian-window SSIM on [0, 255] floats
# protocol='official' : unclamped float PSNR and 8-bit SSIM, as in the released scripts

import math

import cv2
import numpy as np
import torch


def compute_psnr(pred, target, protocol='native'):
    """Mean PSNR (dB) over the batch."""
    if protocol == 'native':
        pred, target = pred.clamp(0, 1), target.clamp(0, 1)
    mse = ((pred.double() - target.double()) ** 2).flatten(1).mean(1)
    return float(torch.where(mse == 0, torch.full_like(mse, 100.0), 10 * torch.log10(1.0 / mse)).mean())


def region_psnr(pred, target, light, bins=3):
    """PSNR inside dark / mid / bright regions of one image.

    light: (1, 1, H, W) illumination prior. Regions are per-image quantiles,
    so each region has the same number of pixels.
    """
    err = ((pred.clamp(0, 1) - target.clamp(0, 1)).double() ** 2).mean(1).flatten()
    lum = light.flatten().double()
    edges = torch.quantile(lum, torch.linspace(0, 1, bins + 1, dtype=lum.dtype, device=lum.device)[1:-1])
    region = torch.bucketize(lum, edges)
    out = []
    for r in range(bins):
        sel = err[region == r]
        mse = float(sel.mean()) if sel.numel() else float('nan')
        out.append(100.0 if mse == 0 else 10 * math.log10(1.0 / mse))
    return out


def compute_ssim(pred, target, protocol='native'):
    return float(np.mean([_ssim(p, t, protocol) for p, t in zip(pred, target)]))


def _ssim(img1, img2, protocol):
    if protocol == 'official':
        img1 = np.uint8((img1.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255).round())
        img2 = np.uint8((img2.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255).round())
    else:
        img1 = img1.detach().clamp(0, 1).cpu().double().numpy().transpose(1, 2, 0) * 255
        img2 = img2.detach().clamp(0, 1).cpu().double().numpy().transpose(1, 2, 0) * 255
    return float(np.mean([_ssim_channel(img1[..., c], img2[..., c]) for c in range(img1.shape[2])]))


def _ssim_channel(img1, img2):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    k = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(k, k.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    s1 = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1 ** 2
    s2 = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2 ** 2
    s12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))
    return ssim_map.mean()
