"""Compare evaluate.py outputs and write markdown tables for the report.

    python compare.py results/official_seed20.json results/baseline_seed20.json results/dwt_seed20.json \
        --out results/comparison.md

The first file is the reference. For every other file we print
  1. PSNR / SSIM per test subset (and the mean over subsets),
  2. the PSNR change in dark / mid / bright regions,
  3. the per-image PSNR change against how uneven the lighting is (light_std quartiles).
If flare suppression helps, the gain should be largest on single_flare and in bright regions.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    data = json.loads(Path(path).read_text())
    return data['config']['name'], data['scores']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('results', nargs='+')
    p.add_argument('--out', default='results/comparison.md')
    args = p.parse_args()

    runs = [load(r) for r in args.results]
    ref_name, ref = runs[0]
    subsets = list(ref)
    lines = ['## PSNR / SSIM per test subset', '',
             '| method | ' + ' | '.join(subsets) + ' | mean |',
             '|---|' + '---:|' * (len(subsets) + 1)]
    rows = list(runs)
    if all('input_psnr' in ref[s] for s in subsets):
        rows.insert(0, ('input (no processing)', {s: {'psnr': ref[s]['input_psnr'], 'ssim': ref[s]['input_ssim']}
                                                  for s in subsets}))
    for name, sc in rows:
        cells = [f"{sc[s]['psnr']:.2f} / {sc[s]['ssim']:.4f}" for s in subsets]
        mean_p = np.mean([sc[s]['psnr'] for s in subsets])
        mean_s = np.mean([sc[s]['ssim'] for s in subsets])
        lines.append(f'| {name} | ' + ' | '.join(cells) + f' | {mean_p:.2f} / {mean_s:.4f} |')

    for name, sc in runs[1:]:
        lines += ['', f'## {name} minus {ref_name}', '',
                  '| subset | dPSNR | dSSIM | dark | mid | bright | images improved | beats input |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        all_gain, all_spread = [], []
        for s in subsets:
            a = {r['name']: r for r in ref[s]['images']}
            b = {r['name']: r for r in sc[s]['images']}
            names = sorted(a.keys() & b.keys())
            gain = np.array([b[n]['psnr'] - a[n]['psnr'] for n in names])
            region = [np.mean([b[n][f'psnr_{k}'] - a[n][f'psnr_{k}'] for n in names])
                      for k in ('dark', 'mid', 'bright')]
            beats = (f"{np.mean([b[n]['psnr'] > b[n]['input_psnr'] for n in names]) * 100:.0f}%"
                     if 'input_psnr' in b[names[0]] else 'n/a')
            all_gain += list(gain)
            all_spread += [a[n]['light_std'] for n in names]
            lines.append(f"| {s} | {gain.mean():+.3f} | {sc[s]['ssim'] - ref[s]['ssim']:+.4f} | "
                         + ' | '.join(f'{r:+.3f}' for r in region)
                         + f' | {(gain > 0).mean() * 100:.0f}% | {beats} |')

        gain, spread = np.array(all_gain), np.array(all_spread)
        edges = np.quantile(spread, [0.25, 0.5, 0.75])
        bins = np.digitize(spread, edges)
        r = np.corrcoef(spread, gain)[0, 1] if len(gain) > 2 else float('nan')
        lines += ['', f'PSNR gain vs lighting unevenness (all test images, Pearson r = {r:+.3f}):', '',
                  '| light_std quartile | images | mean dPSNR |', '|---|---:|---:|']
        labels = ['Q1 (most uniform)', 'Q2', 'Q3', 'Q4 (most uneven)']
        for q, label in enumerate(labels):
            sel = gain[bins == q]
            if sel.size:
                lines.append(f'| {label} | {sel.size} | {sel.mean():+.3f} |')

    text = '\n'.join(lines) + '\n'
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
