"""Evaluation loop shared by validation (train.py) and testing (evaluate.py)."""
from pathlib import Path

import torch
import torchvision.utils as vutils
from torchvision.transforms.functional import resize

from .metrics import compute_psnr, compute_ssim, region_psnr


@torch.no_grad()
def evaluate_weather(net, loader, device, protocol='native', regions=False, save_dir=None):
    """Scores every image in the loader (batch size 1). Returns means and per-image rows."""
    net.eval()
    rows = []
    for inp, gt, imap, rmap, names in loader:
        inp, gt, imap, rmap = [t.to(device) for t in (inp, gt, imap, rmap)]
        if protocol == 'official':
            # released test loader resizes to a multiple of 16 (640x360 -> 640x352)
            size = [gt.shape[-2] // 16 * 16, gt.shape[-1] // 16 * 16]
            inp, gt, imap, rmap = [resize(t, size, antialias=True) for t in (inp, gt, imap, rmap)]
        out = net(inp, imap, rmap)[0]
        row = {'name': names[0],
               'psnr': compute_psnr(out, gt, protocol),
               'ssim': compute_ssim(out, gt, protocol)}
        if regions:
            # input vs GT = the score of doing nothing; a useful model must beat it
            row['input_psnr'] = compute_psnr(inp, gt, protocol)
            row['input_ssim'] = compute_ssim(inp, gt, protocol)
            light = imap.mean(1, keepdim=True)
            row['psnr_dark'], row['psnr_mid'], row['psnr_bright'] = region_psnr(out, gt, light)
            row['light_std'] = float(light.std())  # how uneven the lighting is
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            vutils.save_image(out.clamp(0, 1), Path(save_dir) / names[0])
        rows.append(row)

    summary = {'n': len(rows)}
    for key in rows[0]:
        if key != 'name':
            summary[key] = sum(r[key] for r in rows) / len(rows)
    summary['images'] = rows
    return summary
