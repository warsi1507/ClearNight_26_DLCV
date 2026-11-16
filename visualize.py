"""Save a figure per image: input, illumination prior, flare gate (dwt only), output, GT.

    python visualize.py --checkpoint checkpoints/dwt_seed20/best.pth --subset single_flare --n 6
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from data.dataset import WeatherEvalDataset
from data.splits import branch_paths
from utils.checkpoints import load_checkpoint, model_from_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_root', default='data/AllWeatherNight')
    p.add_argument('--subset', default='single_flare')
    p.add_argument('--n', type=int, default=6)
    p.add_argument('--out_dir', default='results/figures')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = load_checkpoint(args.checkpoint)
    net = model_from_config(ckpt['config']).to(device).eval()
    net.load_state_dict(ckpt['model'])

    data = WeatherEvalDataset(branch_paths(args.data_root, 'test', args.subset))
    out_dir = Path(args.out_dir) / ckpt['config']['name']
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(args.n, len(data))):
        inp, gt, imap, rmap, name = data[i]
        with torch.no_grad():
            out = net(*(t[None].to(device) for t in (inp, imap, rmap)))[0][0].clamp(0, 1).cpu()

        panels = [(inp, 'input'), (imap.mean(0), 'illumination prior')]
        if net.flare is not None:
            panels.append((net.flare.last_gate[0, 0].cpu(), 'flare gate'))
        panels += [(out, 'output'), (gt, 'ground truth')]

        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4))
        for ax, (img, title) in zip(axes, panels):
            if img.dim() == 3:
                ax.imshow(img.permute(1, 2, 0).numpy())
            else:
                ax.imshow(img.numpy(), cmap='magma', vmin=0, vmax=1 if title == 'flare gate' else None)
            ax.set_title(title)
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(out_dir / f'{args.subset}_{Path(name).stem}.png', dpi=120)
        plt.close(fig)
    print(f'saved figures to {out_dir}')


if __name__ == '__main__':
    main()
