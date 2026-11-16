"""Score a trained checkpoint on the seven synthetic AllWeatherNight test subsets.

    python evaluate.py --checkpoint checkpoints/dwt_seed20/best.pth --out_json results/dwt_seed20.json
    python evaluate.py --checkpoint ... --save_images results/images/dwt     # also write restored images

Besides PSNR/SSIM per subset, each image gets PSNR in its dark / mid / bright
thirds (split by the illumination prior) and the spread of its lighting.
compare.py turns two of these json files into the tables for the report.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import WeatherEvalDataset
from data.prepare_dataset import TEST_SUBSETS
from data.splits import branch_paths, load_manifest
from utils.checkpoints import load_checkpoint, model_from_config
from utils.evaluation import evaluate_weather


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_root', default='data/AllWeatherNight')
    p.add_argument('--split', choices=['test', 'val'], default='test')
    p.add_argument('--subsets', nargs='+', default=None)
    p.add_argument('--protocol', choices=['native', 'official'], default='native')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--save_images', default=None, help='folder for restored images')
    p.add_argument('--out_json', default='results/evaluation.json')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = load_checkpoint(args.checkpoint)
    net = model_from_config(ckpt['config']).to(device)
    net.load_state_dict(ckpt['model'])

    if args.split == 'test':
        subsets = args.subsets or list(TEST_SUBSETS)
        sources = {s: (branch_paths(args.data_root, 'test', s), None) for s in subsets}
    else:
        records = load_manifest(Path(args.data_root) / 'split_manifest.json')['records']
        subsets = args.subsets or ['rain', 'snow', 'drop']
        sources = {s: (branch_paths(args.data_root, 'train', s), records[s]['val']) for s in subsets}

    scores = {}
    for subset, (dirs, files) in sources.items():
        loader = DataLoader(WeatherEvalDataset(dirs, files), batch_size=1, num_workers=args.num_workers)
        save_dir = Path(args.save_images) / subset if args.save_images else None
        scores[subset] = evaluate_weather(net, loader, device, args.protocol, regions=True, save_dir=save_dir)
        s = scores[subset]
        print(f"{subset:13s} PSNR {s['psnr']:.4f}  SSIM {s['ssim']:.4f}  "
              f"(input {s['input_psnr']:.2f} / {s['input_ssim']:.4f})  "
              f"(dark/mid/bright {s['psnr_dark']:.2f}/{s['psnr_mid']:.2f}/{s['psnr_bright']:.2f})  n={s['n']}")

    out = {'checkpoint': args.checkpoint, 'config': ckpt['config'], 'epoch': ckpt.get('epoch'),
           'split': args.split, 'protocol': args.protocol, 'scores': scores}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    main()
